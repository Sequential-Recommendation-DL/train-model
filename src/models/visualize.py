# import json
# import os
# from datetime import datetime

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# from torch.utils.data import DataLoader


# def create_result_dir() -> str:
#     """Tạo thư mục results/ngày_giờ/"""
#     timestamp = datetime.now().strftime("%m_%d_%y_%Hh_%Mp")
#     result_dir = os.path.join("results", timestamp)
#     os.makedirs(result_dir, exist_ok=True)
#     return result_dir


# def plot_training_history(history: list[dict], save_dir: str) -> None:
#     """Vẽ biểu đồ Train Loss vs Val Loss theo epoch"""
#     epochs     = [h["epoch"] for h in history]
#     train_loss = [h["train_loss"] for h in history]
#     val_loss   = [h["val_loss"] for h in history]
#     val_acc    = [h["val_acc"] for h in history]

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#     # Loss
#     ax1.plot(epochs, train_loss, marker="o", label="Train Loss", color="steelblue")
#     ax1.plot(epochs, val_loss,   marker="o", label="Val Loss",   color="tomato")
#     ax1.set_title("Train Loss vs Val Loss")
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Loss")
#     ax1.legend()
#     ax1.grid(True)

#     # Accuracy
#     ax2.plot(epochs, val_acc, marker="o", label="Val Accuracy", color="seagreen")
#     ax2.set_title("Val Accuracy theo Epoch")
#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Accuracy")
#     ax2.legend()
#     ax2.grid(True)

#     plt.tight_layout()
#     path = os.path.join(save_dir, "training_history.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  → Đã lưu: {path}")


# def plot_confusion_matrix(
#     model, test_loader: DataLoader, device: str, save_dir: str
# ) -> None:
#     """Vẽ confusion matrix trên test set"""
#     model.eval()
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for X, y in test_loader:
#             X = X.to(device)
#             preds = (model(X) > 0.5).float().squeeze().cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(y.numpy())

#     cm = confusion_matrix(all_labels, all_preds)
#     tn, fp, fn, tp = cm.ravel()

#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
#     plt.colorbar(im, ax=ax)

#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_xticklabels(["Dự đoán: Không thích (0)", "Dự đoán: Thích (1)"])
#     ax.set_yticklabels(["Thực tế: Không thích (0)", "Thực tế: Thích (1)"])

#     for i in range(2):
#         for j in range(2):
#             ax.text(j, i, str(cm[i, j]), ha="center", va="center",
#                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

#     ax.set_title("Confusion Matrix")
#     ax.set_xlabel("Nhãn dự đoán")
#     ax.set_ylabel("Nhãn thực tế")
#     plt.tight_layout()

#     path = os.path.join(save_dir, "confusion_matrix.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  → Đã lưu: {path}")


# def plot_roc_curve(
#     model, test_loader: DataLoader, device: str, save_dir: str
# ) -> None:
#     """Vẽ ROC Curve và tính AUC"""
#     model.eval()
#     all_scores, all_labels = [], []

#     with torch.no_grad():
#         for X, y in test_loader:
#             X = X.to(device)
#             scores = model(X).squeeze().cpu().numpy()
#             all_scores.extend(scores)
#             all_labels.extend(y.numpy())

#     fpr, tpr, _ = roc_curve(all_labels, all_scores)
#     roc_auc = auc(fpr, tpr)

#     fig, ax = plt.subplots(figsize=(6, 5))
#     ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
#     ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
#     ax.set_xlim([0.0, 1.0])
#     ax.set_ylim([0.0, 1.05])
#     ax.set_xlabel("Tỉ lệ dương tính giả (FPR)")
#     ax.set_ylabel("Tỉ lệ dương tính thật (TPR)")
#     ax.set_title("ROC Curve")
#     ax.legend(loc="lower right")
#     ax.grid(True)
#     plt.tight_layout()

#     path = os.path.join(save_dir, "roc_curve.png")
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  → Đã lưu: {path}")

#     return roc_auc


# def save_metrics(metrics: dict, history: list[dict], save_dir: str) -> None:
#     """Lưu metrics ra file JSON"""
#     data = {
#         "test_accuracy": metrics.get("accuracy"),
#         "best_val_acc": max(h["val_acc"] for h in history),
#         "epochs_trained": len(history),
#         "history": history,
#     }
#     path = os.path.join(save_dir, "metrics.json")
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#     print(f"  → Đã lưu: {path}")


# def generate_results(
#     model,
#     history: list[dict],
#     metrics: dict,
#     test_loader: DataLoader,
#     device: str,
# ) -> str:
#     """Hàm tổng hợp — gọi 1 lần trong pipeline"""
#     result_dir = create_result_dir()
#     print(f"\n  Đang tạo kết quả tại: {result_dir}")

#     plot_training_history(history, result_dir)
#     plot_confusion_matrix(model, test_loader, device, result_dir)
#     roc_auc = plot_roc_curve(model, test_loader, device, result_dir)

#     metrics["auc"] = round(float(roc_auc), 4)
#     save_metrics(metrics, history, result_dir)

#     return result_dir
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader


def create_result_dir() -> str:
    """Tạo thư mục results/ngày_giờ/"""
    timestamp = datetime.now().strftime("%m_%d_%y_%Hh_%Mp")
    result_dir = os.path.join("results", timestamp)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def plot_training_history(history: list[dict], save_dir: str) -> None:
    """Vẽ biểu đồ Train Loss vs Val Loss theo epoch"""
    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    val_acc    = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, marker="o", label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_loss,   marker="o", label="Val Loss",   color="tomato")
    ax1.set_title("Train Loss vs Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_acc, marker="o", label="Val Accuracy", color="seagreen")
    ax2.set_title("Val Accuracy theo Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Đã lưu: {path}")


def plot_confusion_matrix(
    model, test_loader: DataLoader, device: str, save_dir: str
) -> None:
    """Vẽ confusion matrix trên test set"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X).squeeze()
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Dự đoán: Không thích (0)", "Dự đoán: Thích (1)"])
    ax.set_yticklabels(["Thực tế: Không thích (0)", "Thực tế: Thích (1)"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    plt.tight_layout()

    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Đã lưu: {path}")


def plot_roc_curve_with_negative_sampling(
    model,
    test_df,
    train_df,
    user_map: dict,
    item_map: dict,
    brand_map: dict,
    device: str,
    save_dir: str,
    num_neg: int = 99,#ban đầu là 99
) -> float:
    """
    Vẽ ROC Curve dùng negative sampling — chuẩn như NeuMF:
    Mỗi user: 1 positive (interaction thực) + 99 negative ngẫu nhiên
    Dùng trực tiếp user_idx, item_idx đã encode sẵn trong df
    """
    model.eval()
    all_scores, all_labels = [], []

    # tập item đã tương tác của mỗi user (từ train) để tránh sample trùng
    user_pos: dict[int, set[int]] = {}
    for _, row in train_df.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        user_pos.setdefault(u, set()).add(i)

    # lấy đúng vocab size từng field từ model
    field_dims = [int(model.offsets[i+1].item()) - int(model.offsets[i].item())
                  for i in range(len(model.offsets)-1)]
    field_dims.append(int(model.embedding.num_embeddings) - int(model.offsets[-1].item()))
    num_users, num_items, num_brands, num_cats, num_prices, num_hours, num_dayofweeks  = field_dims

    # giới hạn số user để ROC nhanh hơn
    max_users = min(2000, len(test_df))
    test_sample = test_df.sample(max_users, random_state=42).reset_index(drop=True)

    # tạo toàn bộ tensor 1 lần, batch forward
    X_list, y_list = [], []
    for _, row in test_sample.iterrows():
        u_idx   = min(int(row["user_idx"]),   num_users  - 1)
        i_idx   = min(int(row["item_idx"]),   num_items  - 1)
        b_idx   = min(int(row.get("brand_idx",    0)), num_brands - 1)
        cat_idx = min(int(row.get("category_idx", 0)), num_cats   - 1)
        p_idx   = min(int(row.get("price_idx",    0)), num_prices - 1)
        h_idx   = min(int(row.get("hour_idx",     0)), num_hours  - 1)
        d_idx   = min(int(row.get("dayofweek_idx", 0)), num_dayofweeks - 1)

        pos_set = user_pos.get(u_idx, set())

        negs = random.sample(
            [j for j in range(num_items) if j != i_idx and j not in pos_set],
            min(num_neg, num_items - len(pos_set) - 1)
        )

        for c, lbl in [(i_idx, 1)] + [(n, 0) for n in negs]:
            X_list.append([u_idx, c, b_idx, cat_idx, p_idx, h_idx, d_idx])
            y_list.append(lbl)

    # batch forward
    batch_size = 4096
    with torch.no_grad():
        for i in range(0, len(X_list), batch_size):
            X = torch.tensor(X_list[i:i+batch_size], dtype=torch.long, device=device)
            scores = torch.sigmoid(model(X).squeeze()).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)
    all_labels = y_list

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tỉ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỉ lệ dương tính thật (TPR)")
    ax.set_title(f"ROC Curve (Negative Sampling, {num_neg} negatives/user)")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()

    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Đã lưu: {path}")

    return roc_auc


def save_metrics(metrics: dict, history: list[dict], save_dir: str) -> None:
    """Lưu metrics ra file JSON"""
    data = {
        "test_accuracy": metrics.get("accuracy"),
        "auc": metrics.get("auc"),
        "best_val_acc": max(h["val_acc"] for h in history),
        "epochs_trained": len(history),
        "history": history,
    }
    path = os.path.join(save_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  → Đã lưu: {path}")


def generate_results(
    model,
    history: list[dict],
    metrics: dict,
    test_loader: DataLoader,
    test_df,
    train_df,
    user_map: dict,
    item_map: dict,
    brand_map: dict,
    device: str,
) -> str:
    """Hàm tổng hợp — gọi 1 lần trong pipeline"""
    result_dir = create_result_dir()
    print(f"\n  Đang tạo kết quả tại: {result_dir}")

    plot_training_history(history, result_dir)
    plot_confusion_matrix(model, test_loader, device, result_dir)
    roc_auc = plot_roc_curve_with_negative_sampling(
        model, test_df, train_df, user_map, item_map, brand_map, device, result_dir
    )

    metrics["auc"] = round(float(roc_auc), 4)
    save_metrics(metrics, history, result_dir)
    print(f"  → AUC (negative sampling): {roc_auc:.4f}")

    return result_dir