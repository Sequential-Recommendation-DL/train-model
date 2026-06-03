import torch
import torch.nn as nn


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_dim: int = 64,
        mlp_dim: int = 64,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 128, 64, 32]

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
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf_out = self.gmf_user(users) * self.gmf_item(items)
        mlp_in = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(fused).squeeze(-1)
