import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class PointwiseDataset(Dataset):
    """Pointwise dataset: each sample is (user, item, label) where label is in [0, 1]."""

    def __init__(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray) -> None:
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.labels = torch.tensor(labels / 2.0, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.labels[idx]


def make_loader(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    device: str,
) -> DataLoader:  # type: ignore[type-arg]
    users = df["user_idx"].to_numpy(dtype=np.int64)
    items = df["item_idx"].to_numpy(dtype=np.int64)
    labels = df["Label"].to_numpy(dtype=np.float32)
    pin = device == "cuda"
    return DataLoader(
        PointwiseDataset(users, items, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin,
    )
