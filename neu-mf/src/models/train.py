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
    all_neg = torch.cat(chunks, dim=0)  # [num_unique_users, top_k]

    user_positives = df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    triplets: list[tuple[int, int, int]] = []
    for i, user_id in enumerate(unique_users.tolist()):
        pos_items = user_positives.get(int(user_id), [])
        neg_items = all_neg[i].tolist()
        for pos in pos_items:
            for neg in neg_items:
                triplets.append((int(user_id), int(pos), int(neg)))

    pin = device == "cuda"
    return DataLoader(  # type: ignore[type-arg]
        BPRDataset(triplets), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin
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

    for epoch in range(1, epochs + 1):
        train_loader = _hard_neg_loader(model, train_df, user_pos, num_items, num_neg, batch_size, device)
        val_loader = _hard_neg_loader(model, val_cap, user_pos, num_items, 4, 1024, device)

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
