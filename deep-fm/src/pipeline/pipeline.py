

# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# from src.models.deepfm import DeepFM, ExplicitDataset
# from src.models.predict import evaluate
# from src.models.save_load import save_model
# from src.models.train import train
# from src.models.visualize import generate_results

# ACC_THRESHOLD = 0.75

# # Output của build.py
# # PROCESSED_TRAIN = "data/processs/train.csv"
# # PROCESSED_VAL   = "data/processs/val.csv"
# PROCESSED_TRAIN = "../data/processs/train.csv"
# PROCESSED_VAL   = "../data/processs/val.csv"


# def negative_sample(df, train_data, num_neg: int = 4):
#     """
#     Negative sampling xen kẽ đúng format:
#     mỗi group = [pos, neg_0, ..., neg_{num_neg-1}]
#     2 fields: user, item (output của build.py)
#     """
#     pos_users = df["UserId"].map(train_data.user_map).fillna(0).astype(int).values
#     pos_items = df["ItemId"].map(train_data.item_map).fillna(0).astype(int).values

#     num_items_vocab = int(train_data.field_dims[1])
#     n = len(df)
#     neg_items = np.random.randint(0, num_items_vocab, size=(n, num_neg))

#     # Xen kẽ: [pos_0, neg_0_0..neg_0_k, pos_1, neg_1_0..neg_1_k, ...]
#     rows_X = np.empty((n * (1 + num_neg), 2), dtype=np.int64)
#     rows_y = np.empty(n * (1 + num_neg), dtype=np.float32)

#     for i in range(n):
#         base = i * (1 + num_neg)
#         rows_X[base] = [pos_users[i], pos_items[i]]
#         rows_y[base] = 1.0
#         for j in range(num_neg):
#             rows_X[base + 1 + j] = [pos_users[i], neg_items[i, j]]
#             rows_y[base + 1 + j] = 0.0

#     X = torch.tensor(rows_X, dtype=torch.long)
#     y = torch.tensor(rows_y, dtype=torch.float32)
#     return TensorDataset(X, y)


# def run_pipeline(
#     model_path: str = "../models/deepfm_best.pth",
#     epochs: int = 10,
#     batch_size: int = 256,
#     lr: float = 0.0001,
#     num_neg_train: int = 4,
#     num_neg_eval: int = 10,
# ) -> dict:
#     import pandas as pd

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")

#     # [1/5] Load data từ build.py output
#     print("\n[1/5] Đọc data từ build.py output...")
#     if not os.path.exists(PROCESSED_TRAIN) or not os.path.exists(PROCESSED_VAL):
#         raise FileNotFoundError(
#             f"Không tìm thấy processed data.\n"
#             f"Chạy trước: python -m data.preprocess.build --rows 250000"
#         )

#     train_df = pd.read_csv(PROCESSED_TRAIN)
#     val_df   = pd.read_csv(PROCESSED_VAL)

#     # Dùng val làm test luôn (build.py chỉ split train/val)
#     test_df  = val_df.copy()

#     num_users = int(max(train_df["UserId"].max(), val_df["UserId"].max())) + 1
#     num_items = int(max(train_df["ItemId"].max(), val_df["ItemId"].max())) + 1
#     print(f"      Train: {len(train_df):,}  Val: {len(val_df):,}")
#     print(f"      Users: {num_users:,}  Items: {num_items:,}")

#     # [2/5] Build dataset & dataloader
#     print("\n[2/5] Building datasets...")
#     train_data = ExplicitDataset(train_df)

#     print(f"      Đang tạo negative samples cho train ({num_neg_train} neg/pos)...")
#     train_dataset = negative_sample(train_df, train_data, num_neg=num_neg_train)

#     print(f"      Đang tạo negative samples cho val ({num_neg_eval} neg/pos)...")
#     val_dataset = negative_sample(val_df, train_data, num_neg=num_neg_eval)

#     print(f"      Đang tạo negative samples cho test ({num_neg_eval} neg/pos)...")
#     test_dataset = negative_sample(test_df, train_data, num_neg=num_neg_eval)

#     print(f"      Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
#     print(f"      field_dims: {train_data.field_dims}")

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
#     val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
#     test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

#     # [3/5] Train
#     print("\n[3/5] Training DeepFM...")
#     field_dims = train_data.field_dims
#     model = DeepFM(field_dims).to(device)
#     model, history = train(
#         model,
#         train_loader,
#         val_loader=val_loader,
#         epochs=epochs,
#         lr=lr,
#         device=device,
#     )

#     # [4/5] Evaluate
#     print("\n[4/5] Evaluating on test set...")
#     metrics = evaluate(model, test_loader, device=device)
#     print(f"      Test Acc: {metrics['accuracy']:.4f}  HR@10: {metrics.get('HR@10', 'N/A')}  NDCG@10: {metrics.get('NDCG@10', 'N/A')}")

#     config = {
#         "field_dims": field_dims.tolist(),
#         "num_users": num_users,
#         "num_items": num_items,
#     }
#     os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
#     save_model(model, model_path, config=config, metadata=metrics)
#     print(f"      Model saved → {model_path}")

#     # [5/5] Generate results
#     print("\n[5/5] Generating results...")
#     result_dir = generate_results(model, history, metrics, test_loader, device)
#     print(f"      Results saved → {result_dir}")

#     if metrics["accuracy"] < ACC_THRESHOLD:
#         print(f"\nWARNING: Accuracy {metrics['accuracy']:.4f} below threshold {ACC_THRESHOLD}.")
#     print(f"      Test Acc: {metrics['accuracy']:.4f}  HR@10: {metrics.get('HR@10', 'N/A')}  NDCG@10: {metrics.get('NDCG@10', 'N/A')}  AUC: {metrics.get('AUC', 'N/A')}")

#     print("\nDone.")
#     return metrics


# if __name__ == "__main__":
#     run_pipeline()

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.deepfm import DeepFM, ExplicitDataset
from src.models.predict import evaluate
from src.models.save_load import save_model
from src.models.train import train
from src.models.visualize import generate_results

ACC_THRESHOLD = 0.75

PROCESSED_TRAIN = "../data/processs/train.csv"
PROCESSED_VAL   = "../data/processs/val.csv"


def negative_sample(df, train_data, num_neg: int = 4):
    """
    Negative sampling xen kẽ đúng format:
    mỗi group = [pos, neg_0, ..., neg_{num_neg-1}]
    3 fields: user, item, category
    """
    pos_users = df["UserId"].map(train_data.user_map).fillna(0).astype(int).values
    pos_items = df["ItemId"].map(train_data.item_map).fillna(0).astype(int).values
    pos_cats  = df["CategoryId"].map(train_data.category_map).fillna(0).astype(int).values

    num_items_vocab = int(train_data.field_dims[1])
    n = len(df)
    neg_items = np.random.randint(0, num_items_vocab, size=(n, num_neg))

    # Xen kẽ: [pos_0, neg_0_0..neg_0_k, pos_1, neg_1_0..neg_1_k, ...]
    rows_X = np.empty((n * (1 + num_neg), 3), dtype=np.int64)
    rows_y = np.empty(n * (1 + num_neg), dtype=np.float32)

    for i in range(n):
        base = i * (1 + num_neg)
        rows_X[base] = [pos_users[i], pos_items[i], pos_cats[i]]
        rows_y[base] = 1.0
        for j in range(num_neg):
            rows_X[base + 1 + j] = [pos_users[i], neg_items[i, j], pos_cats[i]]
            rows_y[base + 1 + j] = 0.0

    X = torch.tensor(rows_X, dtype=torch.long)
    y = torch.tensor(rows_y, dtype=torch.float32)
    return TensorDataset(X, y)


def run_pipeline(
    model_path: str = "../models/deepfm_best.pth",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.0001,
    num_neg_train: int = 4,
    num_neg_eval: int = 10,
) -> dict:
    import pandas as pd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # [1/5] Load data từ build.py output
    print("\n[1/5] Đọc data từ build.py output...")
    if not os.path.exists(PROCESSED_TRAIN) or not os.path.exists(PROCESSED_VAL):
        raise FileNotFoundError(
            f"Không tìm thấy processed data.\n"
            f"Chạy trước: python -m data.preprocess.build --rows 250000"
        )

    train_df = pd.read_csv(PROCESSED_TRAIN)
    val_df   = pd.read_csv(PROCESSED_VAL)
    test_df  = val_df.copy()

    num_users = int(max(train_df["UserId"].max(), val_df["UserId"].max())) + 1
    num_items = int(max(train_df["ItemId"].max(), val_df["ItemId"].max())) + 1
    print(f"      Train: {len(train_df):,}  Val: {len(val_df):,}")
    print(f"      Users: {num_users:,}  Items: {num_items:,}")
    unseen_cats = set(val_df["CategoryId"].unique()) - set(train_df["CategoryId"].unique())
    print(f"      Unseen categories in val: {len(unseen_cats):,}")

    # [2/5] Build dataset & dataloader
    print("\n[2/5] Building datasets...")
    train_data = ExplicitDataset(train_df)

    print(f"      Đang tạo negative samples cho train ({num_neg_train} neg/pos)...")
    train_dataset = negative_sample(train_df, train_data, num_neg=num_neg_train)

    print(f"      Đang tạo negative samples cho val ({num_neg_eval} neg/pos)...")
    val_dataset = negative_sample(val_df, train_data, num_neg=num_neg_eval)

    print(f"      Đang tạo negative samples cho test ({num_neg_eval} neg/pos)...")
    test_dataset = negative_sample(test_df, train_data, num_neg=num_neg_eval)

    print(f"      Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    print(f"      field_dims: {train_data.field_dims}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    # [3/5] Train
    print("\n[3/5] Training DeepFM...")
    field_dims = train_data.field_dims
    model = DeepFM(field_dims).to(device)
    model, history = train(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
    )

    # [4/5] Evaluate
    print("\n[4/5] Evaluating on test set...")
    metrics = evaluate(model, test_loader, device=device)
    print(f"      Test Acc: {metrics['accuracy']:.4f}  HR@10: {metrics.get('HR@10', 'N/A')}  NDCG@10: {metrics.get('NDCG@10', 'N/A')}  AUC: {metrics.get('AUC', 'N/A')}")

    config = {
        "field_dims": field_dims.tolist(),
        "num_users": num_users,
        "num_items": num_items,
    }
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    save_model(model, model_path, config=config, metadata=metrics)
    print(f"      Model saved → {model_path}")

    # [5/5] Generate results
    print("\n[5/5] Generating results...")
    result_dir = generate_results(model, history, metrics, test_loader, device)
    print(f"      Results saved → {result_dir}")

    if metrics["accuracy"] < ACC_THRESHOLD:
        print(f"\nWARNING: Accuracy {metrics['accuracy']:.4f} below threshold {ACC_THRESHOLD}.")

    print("\nDone.")
    return metrics


if __name__ == "__main__":
    run_pipeline()