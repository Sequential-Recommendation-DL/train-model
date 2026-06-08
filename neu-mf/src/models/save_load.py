import os

import torch

from src.models.neumf import NeuMF


def save_model(
    model: NeuMF,
    path: str,
    config: dict,  # type: ignore[type-arg]
    metadata: dict | None = None,  # type: ignore[type-arg]
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": config, "metadata": metadata or {}},
        path,
    )
    print(f"Model saved → {path}")


def load_model(path: str) -> tuple[NeuMF, dict]:  # type: ignore[type-arg]
    checkpoint = torch.load(path, weights_only=False)
    model = NeuMF(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["metadata"]
