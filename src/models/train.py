import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: str = "models/deepfm_best.pth",
):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = (model(X) > 0.5).float().squeeze()
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}  Avg Loss: {avg_loss:.4f}  Val Acc: {val_acc:.4f}")

        history.append({"epoch": epoch + 1, "loss": avg_loss, "val_acc": val_acc})

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  → Best model saved (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    return model, history


if __name__ == "__main__":
    # chạy trực tiếp: python -m src.models.train
    # Khuyến khích chạy qua pipeline.py thay vì file này
    print("Run pipeline.py instead: python -m src.pipeline.pipeline")