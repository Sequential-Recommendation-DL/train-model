import os
import torch
import numpy as np
from src.models.deepfm import DeepFM


def save_model(
    model: DeepFM,
    path: str,
    config: dict,
    metadata: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": config, "metadata": metadata or {}},
        path,
    )
    print(f"Model saved → {path}")


def load_model(path: str) -> tuple[DeepFM, dict]:
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint["config"]

    # field_dims được lưu dạng list, cần convert lại numpy array
    config["field_dims"] = np.array(config["field_dims"])

    model = DeepFM(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["metadata"]