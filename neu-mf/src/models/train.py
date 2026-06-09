import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.features.build_features import BPRDataset, get_hard_negatives
from src.models.neumf import NeuMF
from src.models.predict import evaluate

_CHUNK = 512  # users processed per get_hard_negatives call to bound peak memory


def _hard_neg_loader(
    model: NeuMF,
    df: pd.DataFrame,
    user_pos: dict[int, set[int]],
    num_items: int,
    num_neg: int,
    batch_size: int,
    device: str,
) -> DataLoader:  # type: ignore[type-arg]
    unique_users = np.asarray(df["user_idx"].unique(), dtype=np.int64)
    top_k = min(num_neg, num_items - 1)
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        user_emb = model.gmf_user.weight.detach()
        item_emb = model.gmf_item.weight.detach().to(device)
        for start in range(0, len(unique_users), _CHUNK):
            chunk = unique_users[start : start + _CHUNK].tolist()
            neg_idx = get_hard_negatives(
                user_emb[chunk].to(device), item_emb, chunk, user_pos, top_k
            )
            chunks.append(neg_idx.cpu())
    all_neg = torch.cat(chunks, dim=0).numpy()  # [num_unique_users, top_k]

    pair_users = df["user_idx"].to_numpy(dtype=np.int64)
    pair_items = df["item_idx"].to_numpy(dtype=np.int64)

    # Vectorized: map each pair's user_id → its row index in unique_users
    uid_to_row = np.full(int(unique_users.max()) + 1, -1, dtype=np.int64)
    uid_to_row[unique_users] = np.arange(len(unique_users), dtype=np.int64)
    row_indices = uid_to_row[pair_users]          # [N]

    neg_for_pairs = all_neg[row_indices]          # [N, top_k]
    users_rep = np.repeat(pair_users, top_k)      # [N * top_k]
    items_rep = np.repeat(pair_items, top_k)      # [N * top_k]
    negs_flat = neg_for_pairs.reshape(-1)         # [N * top_k]

    triplets = np.stack([users_rep, items_rep, negs_flat], axis=1)

    pin = device == "cuda"
    return DataLoader(  # type: ignore[type-arg]
        BPRDataset(triplets), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin
    )


def _random_neg_loader(
    df: pd.DataFrame,
    user_pos: dict[int, set[int]],
    num_items: int,
    num_neg: int,
    batch_size: int,
    device: str,
    seed: int = 42,
) -> DataLoader:  # type: ignore[type-arg]
    """Val loader with random negatives — gives a stable loss signal independent of model state."""
    rng = np.random.default_rng(seed)
    pair_users = df["user_idx"].to_numpy(dtype=np.int64)
    pair_items = df["item_idx"].to_numpy(dtype=np.int64)
    n = len(pair_users)

    candidates = rng.integers(0, num_items, size=(n, num_neg * 4))
    rows: list[tuple[int, int, int]] = []
    for i in range(n):
        u, p = int(pair_users[i]), int(pair_items[i])
        pos_set = user_pos.get(u, set())
        negs = [int(c) for c in candidates[i] if c not in pos_set][:num_neg]
        for neg in negs:
            rows.append((u, p, neg))

    pin = device == "cuda"
    return DataLoader(  # type: ignore[type-arg]
        BPRDataset(rows), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )


def train(
    model: NeuMF,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user_pos: dict[int, set[int]],
    num_items: int,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    device: str = "cpu",
    max_val_users: int = 5_000,
    num_neg: int = 4,
    batch_size: int = 512,
    hard_neg_freq: int = 3,
) -> tuple[NeuMF, list[dict]]:  # type: ignore[type-arg]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    best_hr = 0.0
    no_improve = 0
    best_state: dict | None = None  # type: ignore[type-arg]
    history: list[dict] = []  # type: ignore[type-arg]

    val_cap = val_df if max_val_users is None else val_df.sample(
        min(max_val_users, len(val_df)), random_state=42
    ).reset_index(drop=True)

    # Val loader uses fixed random negatives (built once) so val_loss is a stable generalization signal.
    # Hard-mined val negatives collapse to 0 as training converges — not informative.
    val_loader = _random_neg_loader(val_cap, user_pos, num_items, 4, 1024, device)
    train_loader: DataLoader | None = None  # type: ignore[type-arg]

    for epoch in range(1, epochs + 1):
        # Refresh hard negatives on epoch 1 and every hard_neg_freq epochs thereafter
        if (epoch - 1) % hard_neg_freq == 0:
            train_loader = _hard_neg_loader(model, train_df, user_pos, num_items, num_neg, batch_size, device)
        assert train_loader is not None

        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", unit="batch", leave=False) as pbar:
            for users, pos_items, neg_items in pbar:
                users = users.to(device)
                pos_items = pos_items.to(device)
                neg_items = neg_items.to(device)
                optimizer.zero_grad()
                loss = -F.logsigmoid(model(users, pos_items) - model(users, neg_items)).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for users, pos_items, neg_items in val_loader:
                users = users.to(device)
                pos_items = pos_items.to(device)
                neg_items = neg_items.to(device)
                val_total += (-F.logsigmoid(model(users, pos_items) - model(users, neg_items)).mean()).item()
        val_loss = val_total / len(val_loader)

        metrics = evaluate(
            model, val_df, user_pos, num_items,
            device=device, max_users=max_val_users,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history.append({"epoch": epoch, "loss": avg_loss, "val_loss": val_loss, "lr": current_lr, **metrics})
        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_loss={val_loss:.4f}"
            f" | HR@10={metrics['HR@10']:.4f} | NDCG@10={metrics['NDCG@10']:.4f}"
            f" | lr={current_lr:.2e}"
        )

        if metrics["HR@10"] > best_hr:
            best_hr = metrics["HR@10"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stop at epoch {epoch}: HR@10 has not improved for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
