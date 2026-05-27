import copy

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for users, items, labels in train_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(users, items), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(
            model, val_df, user_pos, num_items,
            device=device, max_users=max_val_users,
        )
        history.append({"epoch": epoch, "loss": total_loss, **metrics})
        print(
            f"Epoch {epoch:02d} | loss={total_loss:.4f}"
            f" | HR@10={metrics['HR@10']:.4f} | NDCG@10={metrics['NDCG@10']:.4f}"
        )

        if metrics["HR@10"] > best_hr:
            best_hr = metrics["HR@10"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
