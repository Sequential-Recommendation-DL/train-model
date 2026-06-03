import torch
from torch.utils.data import DataLoader
from src.models.deepfm import DeepFM


def evaluate(
    model: DeepFM,
    test_loader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = (model(X) > 0.5).float().squeeze()
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}