import math
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

    rng = np.random.default_rng(42)
    all_users: list[int] = []
    all_items: list[int] = []

    for _, row in test_df.iterrows():
        u, pos = int(row["user_idx"]), int(row["item_idx"])  # type: ignore[arg-type]
        pos_set = user_pos.get(u, set())
        pool = rng.integers(0, num_items, size=num_neg * 4).tolist()
        negs: list[int] = [j for j in pool if j != pos and j not in pos_set][:num_neg]
        while len(negs) < num_neg:
            j = int(rng.integers(0, num_items))
            if j != pos and j not in pos_set:
                negs.append(j)
        all_users.extend([u] * (num_neg + 1))
        all_items.extend([pos] + negs)

    users_t = torch.tensor(all_users, dtype=torch.long)
    items_t = torch.tensor(all_items, dtype=torch.long)

    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(users_t), 65536):
            chunks.append(
                model(users_t[i : i + 65536].to(device), items_t[i : i + 65536].to(device)).cpu()
            )
    preds_mat = torch.cat(chunks).numpy().reshape(len(test_df), num_neg + 1)

    # Positive is always at column 0; count items that score higher to get 1-based rank
    ranks = (preds_mat > preds_mat[:, :1]).sum(axis=1) + 1
    in_top_k = ranks <= k
    return {
        f"HR@{k}": float(in_top_k.mean()),
        f"NDCG@{k}": float(np.where(in_top_k, 1.0 / np.log2(ranks + 1), 0.0).mean()),
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

    rng = np.random.default_rng(42)
    all_users: list[int] = []
    all_items: list[int] = []

    for _, row in test_df.iterrows():
        u, pos = int(row["user_idx"]), int(row["item_idx"])  # type: ignore[arg-type]
        pos_set = user_pos.get(u, set())
        pool = rng.integers(0, num_items, size=num_neg * 4).tolist()
        negs: list[int] = [j for j in pool if j != pos and j not in pos_set][:num_neg]
        while len(negs) < num_neg:
            j = int(rng.integers(0, num_items))
            if j != pos and j not in pos_set:
                negs.append(j)
        all_users.extend([u] * (num_neg + 1))
        all_items.extend([pos] + negs)

    users_t = torch.tensor(all_users, dtype=torch.long)
    items_t = torch.tensor(all_items, dtype=torch.long)

    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(users_t), 65536):
            chunks.append(
                model(users_t[i : i + 65536].to(device), items_t[i : i + 65536].to(device)).cpu()
            )
    all_preds = torch.cat(chunks).numpy()

    n = len(test_df)
    stride = num_neg + 1
    preds_mat = all_preds.reshape(n, stride)
    ranks = (preds_mat > preds_mat[:, :1]).sum(axis=1) + 1
    in_top_k = ranks <= k
    hr = float(in_top_k.mean())
    ndcg = float(np.where(in_top_k, 1.0 / np.log2(ranks + 1), 0.0).mean())

    labels_arr = np.tile([1] + [0] * num_neg, n).astype(np.int32)
    probs = (1.0 / (1.0 + np.exp(-all_preds))).astype(np.float32)
    auc = float(roc_auc_score(labels_arr, probs))

    # Ranking-based CM: predict top-K per user group as positive.
    # Recall = HR@K (a hit = TP), so this CM is the most meaningful for ranking models.
    preds_ranking = np.zeros(len(labels_arr), dtype=np.int32)
    for i in range(n):
        start = i * stride
        top_k_idx = np.argsort(all_preds[start : start + stride])[::-1][:k]
        preds_ranking[start + top_k_idx] = 1
    cm_ranking = confusion_matrix(labels_arr, preds_ranking)

    # Optimal-threshold CM: threshold that maximises Youden's J (sensitivity + specificity - 1).
    from sklearn.metrics import roc_curve as _roc_curve
    fpr_arr, tpr_arr, thr_arr = _roc_curve(labels_arr, probs)
    opt_thr = float(thr_arr[np.argmax(tpr_arr - fpr_arr)])
    preds_opt = (probs >= opt_thr).astype(np.int32)
    cm_opt = confusion_matrix(labels_arr, preds_opt)

    # Fixed-threshold CM kept as supplementary reference.
    preds_t05 = (probs >= 0.5).astype(np.int32)
    cm_t05 = confusion_matrix(labels_arr, preds_t05)

    return {
        f"HR@{k}": hr,
        f"NDCG@{k}": ndcg,
        "n_hits_at_k": int(in_top_k.sum()),
        "n_eval_users": n,
        "AUC_ROC": auc,
        # Ranking-based metrics (primary — preds_ranking so Recall == HR@K)
        "Precision": float(precision_score(labels_arr, preds_ranking, zero_division=0)),  # type: ignore[arg-type]
        "Recall": float(recall_score(labels_arr, preds_ranking, zero_division=0)),  # type: ignore[arg-type]
        "F1": float(f1_score(labels_arr, preds_ranking, zero_division=0)),  # type: ignore[arg-type]
        "confusion_matrix": cm_ranking.tolist(),
        # Optimal-threshold metrics (supplementary)
        "optimal_threshold": opt_thr,
        "confusion_matrix_opt": cm_opt.tolist(),
        # Fixed-threshold reference (0.5 — misleading for ranking models, kept for comparison)
        "confusion_matrix_t05": cm_t05.tolist(),
        "_raw_scores": all_preds.tolist(),
        "_raw_labels": labels_arr.tolist(),
    }
