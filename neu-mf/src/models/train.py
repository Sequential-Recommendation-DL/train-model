import copy

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.features.build_features import make_loader
from src.models.neumf import NeuMF


def train(
    model: NeuMF,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[NeuMF, list[dict]]:  # type: ignore[type-arg]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = make_loader(train_df, batch_size, shuffle=True, device=device)
    val_loader = make_loader(val_df, batch_size, shuffle=False, device=device)

    best_val_loss = float("inf")
    no_improve = 0
    best_state: dict | None = None  # type: ignore[type-arg]
    history: list[dict] = []  # type: ignore[type-arg]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", unit="batch", leave=False) as pbar:
            for users, items, labels in pbar:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(users, items), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                val_total += criterion(model(users, items), labels).item()
        avg_val_loss = val_total / len(val_loader)

        history.append({"epoch": epoch, "loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"Epoch {epoch:02d} | loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stop at epoch {epoch}: val_loss has not improved for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
