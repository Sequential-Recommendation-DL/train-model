from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_user_pos(df: pd.DataFrame) -> dict[int, set[int]]:
    user_pos: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(df["user_idx"].to_numpy(), df["item_idx"].to_numpy()):
        user_pos[int(u)].add(int(i))
    return dict(user_pos)


def negative_sample(
    df: pd.DataFrame,
    num_items: int,
    user_pos: dict[int, set[int]],
    num_neg: int = 4,
) -> pd.DataFrame:
    rng = np.random.default_rng()
    users: np.ndarray = df["user_idx"].to_numpy(dtype=np.int64)
    items: np.ndarray = df["item_idx"].to_numpy(dtype=np.int64)
    n = len(users)

    neg_users: np.ndarray = np.repeat(users, num_neg)
    neg_items: np.ndarray = rng.integers(0, num_items, size=n * num_neg)

    # Rejection sampling for collisions — rate is low when num_items >> user history size
    for idx in range(len(neg_users)):
        u = int(neg_users[idx])
        pos_set = user_pos.get(u, set())
        while int(neg_items[idx]) in pos_set:
            neg_items[idx] = rng.integers(0, num_items)

    return pd.DataFrame({
        "user_idx": np.concatenate([users, neg_users]),
        "item_idx": np.concatenate([items, neg_items]),
        "label": np.concatenate([
            np.ones(n, dtype=np.float32),
            np.zeros(n * num_neg, dtype=np.float32),
        ]),
    })


class NCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.users = torch.tensor(df["user_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.labels = torch.tensor(df["label"].to_numpy(dtype=np.float32), dtype=torch.float)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.labels[idx]
