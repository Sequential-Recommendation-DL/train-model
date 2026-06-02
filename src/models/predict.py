import math
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.neumf import NeuMF


def hit_rate_at_k(scores: list[tuple[float, int]], k: int = 10) -> float:
    top_k = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
    return float(any(label == 1 for _, label in top_k))


def ndcg_at_k(scores: list[tuple[float, int]], k: int = 10) -> float:
    top_k = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
    for rank, (_, label) in enumerate(top_k, start=1):
        if label == 1:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def evaluate(
    model: NeuMF,
    test_df: pd.DataFrame,
    user_pos: dict[int, set[int]],
    num_items: int,
    k: int = 10,
    num_neg: int = 99,
    device: str = "cpu",
    max_users: int | None = None,
) -> dict[str, float]:
    if max_users is not None and len(test_df) > max_users:
        test_df = test_df.sample(max_users, random_state=42).reset_index(drop=True)

    model.eval()
    hr_list: list[float] = []
    ndcg_list: list[float] = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            u = int(row["user_idx"])
            pos = int(row["item_idx"])
            pos_set = user_pos.get(u, set())

            # Over-sample candidates to minimise rejection loop iterations
            pool = np.random.randint(0, num_items, size=num_neg * 3).tolist()
            negs: list[int] = [j for j in pool if j != pos and j not in pos_set][:num_neg]
            while len(negs) < num_neg:
                j = random.randint(0, num_items - 1)
                if j != pos and j not in pos_set:
                    negs.append(j)

            candidates = [pos] + negs
            users_t = torch.tensor([u] * len(candidates), dtype=torch.long, device=device)
            items_t = torch.tensor(candidates, dtype=torch.long, device=device)
            preds = model(users_t, items_t).cpu().tolist()
            scored = list(zip(preds, [1] + [0] * num_neg))
            hr_list.append(hit_rate_at_k(scored, k))
            ndcg_list.append(ndcg_at_k(scored, k))

    return {
        f"HR@{k}": sum(hr_list) / len(hr_list),
        f"NDCG@{k}": sum(ndcg_list) / len(ndcg_list),
    }


def evaluate_full(
    model: NeuMF,
    test_df: pd.DataFrame,
    user_pos: dict[int, set[int]],
    num_items: int,
    k: int = 10,
    num_neg: int = 99,
    device: str = "cpu",
    max_users: int | None = None,
) -> dict[str, Any]:
    """Ranking metrics (HR@K, NDCG@K) + classification metrics (AUC-ROC, Precision, Recall, F1, confusion matrix)."""
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if max_users is not None and len(test_df) > max_users:
        test_df = test_df.sample(max_users, random_state=42).reset_index(drop=True)

    model.eval()
    hr_list: list[float] = []
    ndcg_list: list[float] = []
    all_scores: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            u = int(row["user_idx"])
            pos = int(row["item_idx"])
            pos_set = user_pos.get(u, set())

            pool = np.random.randint(0, num_items, size=num_neg * 3).tolist()
            negs: list[int] = [j for j in pool if j != pos and j not in pos_set][:num_neg]
            while len(negs) < num_neg:
                j = random.randint(0, num_items - 1)
                if j != pos and j not in pos_set:
                    negs.append(j)

            candidates = [pos] + negs
            users_t = torch.tensor([u] * len(candidates), dtype=torch.long, device=device)
            items_t = torch.tensor(candidates, dtype=torch.long, device=device)
            preds = model(users_t, items_t).cpu().tolist()
            scored = list(zip(preds, [1] + [0] * num_neg))

            hr_list.append(hit_rate_at_k(scored, k))
            ndcg_list.append(ndcg_at_k(scored, k))
            all_scores.extend(preds)
            all_labels.extend([1] + [0] * num_neg)

    scores_arr = np.array(all_scores, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int32)
    probs = 1.0 / (1.0 + np.exp(-scores_arr))  # sigmoid
    preds_bin = (probs >= 0.5).astype(int)

    cm = confusion_matrix(labels_arr, preds_bin)
    auc = float(roc_auc_score(labels_arr, probs))
    precision = float(precision_score(labels_arr, preds_bin, zero_division=0))
    recall = float(recall_score(labels_arr, preds_bin, zero_division=0))
    f1 = float(f1_score(labels_arr, preds_bin, zero_division=0))

    return {
        f"HR@{k}": sum(hr_list) / len(hr_list),
        f"NDCG@{k}": sum(ndcg_list) / len(ndcg_list),
        "AUC_ROC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "confusion_matrix": cm.tolist(),
        "_raw_scores": scores_arr.tolist(),
        "_raw_labels": labels_arr.tolist(),
    }
