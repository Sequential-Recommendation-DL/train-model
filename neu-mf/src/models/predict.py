import numpy as np
import pandas as pd
import torch

from src.models.neumf import NeuMF


def evaluate(model: NeuMF, val_df: pd.DataFrame, device: str = "cpu") -> dict:  # type: ignore[type-arg]
    from sklearn.metrics import roc_auc_score

    users = torch.tensor(val_df["user_idx"].to_numpy(dtype=np.int64))
    items = torch.tensor(val_df["item_idx"].to_numpy(dtype=np.int64))
    labels_norm = val_df["Label"].to_numpy(dtype=np.float32) / 2.0

    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(users), 65536):
            chunks.append(
                model(users[i : i + 65536].to(device), items[i : i + 65536].to(device)).cpu()
            )
    logits = torch.cat(chunks).numpy()
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    # Binarise at 0.5 of the normalised label (raw Label > 1.0)
    binary_labels = (labels_norm > 0.5).astype(np.int32)
    auc = float(roc_auc_score(binary_labels, probs)) if len(np.unique(binary_labels)) > 1 else 0.0

    return {
        "AUC_ROC": auc,
        "_raw_scores": logits.tolist(),
        "_raw_labels": binary_labels.tolist(),
    }
