import torch
import torch.nn as nn


def _mlp_layer_sizes(mlp_dim: int, min_hidden: int = 8) -> list[int]:
    """Compute a halving funnel from mlp_dim down to min_hidden.

    MLP input is mlp_dim*2 (user_emb || item_emb), so the first hidden
    layer starts at mlp_dim (half of input) and keeps halving until the
    layer width would drop below min_hidden.

    Example: mlp_dim=64, min_hidden=8 → [64, 32, 16, 8]
    """
    sizes, size = [], mlp_dim
    while size >= min_hidden:
        sizes.append(size)
        size //= 2
    return sizes


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_dim: int = 64,
        mlp_dim: int = 64,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if mlp_layers is None:
            mlp_layers = _mlp_layer_sizes(mlp_dim)

        self.gmf_user = nn.Embedding(num_users, gmf_dim)
        self.gmf_item = nn.Embedding(num_items, gmf_dim)
        self.mlp_user = nn.Embedding(num_users, mlp_dim)
        self.mlp_item = nn.Embedding(num_items, mlp_dim)

        in_dim = mlp_dim * 2
        layers: list[nn.Module] = []
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(gmf_dim + mlp_layers[-1], 1)
        self._mlp_input_dim = mlp_dim * 2
        self._mlp_layers = mlp_layers
        self._gmf_dim = gmf_dim
        self._dropout = dropout
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.05)
        # Zero-init all linear biases so embedding signal is not drowned out by
        # large random bias terms that are shared across all users/items.
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def param_summary(self) -> str:
        emb_params = sum(
            p.numel() for m in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item)
            for p in m.parameters()
        )
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        out_params = sum(p.numel() for p in self.output.parameters())
        total = emb_params + mlp_params + out_params

        shape = " -> ".join(
            [str(self._mlp_input_dim)] + [str(d) for d in self._mlp_layers] + ["1"]
        )
        return (
            f"NeuMF parameter summary:\n"
            f"  Embeddings : {emb_params:>12,}\n"
            f"  MLP layers : {mlp_params:>12,}\n"
            f"  Output     : {out_params:>12,}\n"
            f"  Total      : {total:>12,}\n"
            f"  MLP shape  : {shape}"
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf_out = self.gmf_user(users) * self.gmf_item(items)
        mlp_in = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(fused).squeeze(-1)
