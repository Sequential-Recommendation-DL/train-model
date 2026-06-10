
# import torch
# from torch.utils.data import DataLoader
# from src.models.deepfm import DeepFM


# def evaluate(
#     model: DeepFM,
#     test_loader: DataLoader,
#     device: str = "cpu",
# ) -> dict[str, float]:
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for X, y in test_loader:
#             X, y = X.to(device), y.to(device)
#             # model giờ trả logits → cần sigmoid trước khi threshold
#             logits = model(X).squeeze()
#             preds = (torch.sigmoid(logits) > 0.5).float()
#             correct += (preds == y).sum().item()
#             total += y.size(0)

#     accuracy = correct / total if total > 0 else 0.0
#     return {"accuracy": accuracy}
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score  


def evaluate(
    model,
    test_loader: DataLoader,
    device: str = "cpu",
    k: int = 10,
) -> dict[str, float]:
    """
    Đánh giá model bằng HR@K và NDCG@K.
    test_loader phải chứa TensorDataset với negative sampling
    (1 positive + num_neg negative mỗi user).
    """
    model.eval()
    
    # tính accuracy đơn giản
    correct, total = 0, 0
    all_scores, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze()
            preds  = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total   += y.size(0)
            all_scores.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    accuracy = correct / total if total > 0 else 0.0

    # tính HR@K và NDCG@K
    # giả sử mỗi nhóm = 1 positive + num_neg negative liên tiếp
    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    # tìm num_neg tự động: mỗi group bắt đầu bằng positive
    # tìm khoảng cách giữa các positive
    pos_indices = np.where(labels_arr == 1)[0]
    if len(pos_indices) < 2:
        return {"accuracy": accuracy, "HR@10": 0.0, "NDCG@10": 0.0}

    group_size = int(pos_indices[1] - pos_indices[0])

    hr_list, ndcg_list = [], []

    for start in pos_indices:
        end = start + group_size
        if end > len(scores_arr):
            break

        group_scores = scores_arr[start:end]
        group_labels = labels_arr[start:end]

        # xếp hạng theo score giảm dần
        ranked = np.argsort(group_scores)[::-1]
        top_k  = ranked[:k]

        # HR@K: positive có trong top K không
        hit = int(group_labels[top_k].sum() > 0)
        hr_list.append(hit)

        # NDCG@K: vị trí của positive trong top K
        ndcg = 0.0
        for rank, idx in enumerate(top_k):
            if group_labels[idx] == 1:
                ndcg = 1.0 / math.log2(rank + 2)
                break
        ndcg_list.append(ndcg)

    hr   = sum(hr_list)   / len(hr_list)   if hr_list   else 0.0
    ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = 0.0

    return {
        "accuracy": accuracy,
        f"HR@{k}":   round(hr,   4),
        f"NDCG@{k}": round(ndcg, 4),
        "AUC":       round(auc,  4),
    }