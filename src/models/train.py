import copy

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.features.build_features import NCFDataset, negative_sample
from src.models.neumf import NeuMF
from src.models.predict import evaluate


def train(
    model: NeuMF,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_df: pd.DataFrame,
    user_pos: dict[int, set[int]],
    num_items: int,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    max_val_users: int = 5_000,
) -> tuple[NeuMF, list[dict]]:  # type: ignore[type-arg]
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_hr = 0.0
    best_state: dict | None = None  # type: ignore[type-arg]
    history: list[dict] = []  # type: ignore[type-arg]

    # Pre-generate fixed val negatives once for consistent loss tracking across epochs
    val_cap = val_df if max_val_users is None else val_df.sample(
        min(max_val_users, len(val_df)), random_state=42
    ).reset_index(drop=True)
    val_neg = negative_sample(val_cap, num_items, user_pos, num_neg=4)
    val_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        NCFDataset(val_neg), batch_size=1024, shuffle=False, num_workers=0
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", unit="batch", leave=False) as pbar:
            for users, items, labels in pbar:
                users = users.to(device)
                items = items.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(users, items), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for users, items, labels in val_loader:
                users = users.to(device)
                items = items.to(device)
                labels = labels.to(device)
                val_total += criterion(model(users, items), labels).item()
        val_loss = val_total / len(val_loader)

        metrics = evaluate(
            model, val_df, user_pos, num_items,
            device=device, max_users=max_val_users,
        )
        history.append({"epoch": epoch, "loss": avg_loss, "val_loss": val_loss, **metrics})
        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_loss={val_loss:.4f}"
            f" | HR@10={metrics['HR@10']:.4f} | NDCG@10={metrics['NDCG@10']:.4f}"
        )

        if metrics["HR@10"] > best_hr:
            best_hr = metrics["HR@10"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
