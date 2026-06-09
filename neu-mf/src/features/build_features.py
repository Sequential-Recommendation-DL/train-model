from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import Tensor


def build_user_pos(df):
    user_pos = defaultdict(set)
    for u, i in zip(df["user_idx"].to_numpy(), df["item_idx"].to_numpy()):
        user_pos[int(u)].add(int(i))
    return dict(user_pos)

def get_hard_negatives(
    user_embeddings: Tensor,
    item_embeddings: Tensor,
    batch_user_ids: list,
    user_pos_dict: dict,
    top_k: int = 4
) -> Tensor:
    with torch.no_grad():
        # 1. Matrix Multiplication [batch_size, num_items]
        scores: Tensor = torch.matmul(user_embeddings, item_embeddings.T)

        num_items = item_embeddings.size(0)

        # 2. Tạo mask loại bỏ các positive items
        mask: Tensor = torch.zeros(scores.shape, device=scores.device).bool()

        for i, user_id in enumerate(batch_user_ids):
            pos_items = list(user_pos_dict.get(user_id, []))
            if pos_items:
                # Lọc index hợp lệ, tránh IndexError
                valid_pos = [idx for idx in pos_items if 0 <= idx < num_items]
                if valid_pos:
                    mask[i, valid_pos] = True

        # 3. Mask positive items
        scores = scores.masked_fill(mask, -1e9)

        # 4. Clamp top_k so we never return masked positives
        available = int((~mask).sum(dim=1).min().item())
        top_k = min(top_k, available)

        _, hard_neg_indices = torch.topk(scores, top_k, dim=1)

    return hard_neg_indices

class NCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.users = torch.tensor(df["user_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
        self.labels = torch.tensor(df["label"].to_numpy(dtype=np.float32), dtype=torch.float)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.labels[idx]


class BPRDataset(Dataset):
    """Pairwise dataset for BPR loss: each sample is (user, pos_item, neg_item)."""

    def __init__(self, triplets: list[tuple[int, int, int]] | np.ndarray) -> None:
        arr = np.array(triplets, dtype=np.int64)
        self.users = torch.tensor(arr[:, 0], dtype=torch.long)
        self.pos_items = torch.tensor(arr[:, 1], dtype=torch.long)
        self.neg_items = torch.tensor(arr[:, 2], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]
