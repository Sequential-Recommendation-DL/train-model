
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_all
from src.data.preprocess import clean, encode, split, validate
from src.models.deepfm import DeepFM, ExplicitDataset
from src.models.predict import evaluate
from src.models.save_load import save_model
from src.models.train import train
from src.models.visualize import generate_results

ACC_THRESHOLD = 0.75


def negative_sample(df, train_data, num_neg: int = 4):
    """
    Tạo negative samples cho một tập dữ liệu.
    
    Với mỗi positive (user, item), sample num_neg item ngẫu nhiên
    mà user chưa từng tương tác → label = 0.
    
    Args:
        df: DataFrame chứa positive interactions
        train_data: ExplicitDataset của train (để lấy map và field_dims)
        num_neg: số negative mỗi positive
    
    Returns:
        TensorDataset gồm (X, y) đã có cả positive lẫn negative
    """
    # build tập items đã tương tác của từng user để tránh sample nhầm
    user_pos_items = df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    num_items_vocab = int(train_data.field_dims[1])
    n = len(df)

    pos_users      = df["user_idx"].map(train_data.user_map).fillna(0).astype(int).values
    pos_items      = df["item_idx"].map(train_data.item_map).fillna(0).astype(int).values
    pos_brands     = df["brand_idx"].map(train_data.brand_map).fillna(0).astype(int).values
    pos_cats       = df["category_idx"].map(train_data.category_map).fillna(0).astype(int).values
    pos_prices     = df["price_idx"].fillna(0).astype(int).values
    # [THÊM] hour và dayofweek giữ nguyên của positive (negative cùng context thời gian)
    pos_hours      = df["hour_idx"].fillna(0).astype(int).values
    pos_dayofweeks = df["dayofweek_idx"].fillna(0).astype(int).values

    # sample negative items (tránh item đã tương tác)
    neg_items = np.zeros((n, num_neg), dtype=int)
    for i, (user_idx, row_items) in enumerate(zip(df["user_idx"].values, [user_pos_items.get(u, set()) for u in df["user_idx"].values])):
        sampled = []
        while len(sampled) < num_neg:
            candidates = np.random.randint(0, num_items_vocab, size=num_neg * 2)
            for c in candidates:
                if c not in row_items and len(sampled) < num_neg:
                    sampled.append(c)
        neg_items[i] = sampled

    # ghép positive và negative
    all_users      = np.concatenate([pos_users,      np.repeat(pos_users,      num_neg)])
    all_items      = np.concatenate([pos_items,      neg_items.flatten()])
    all_brands     = np.concatenate([pos_brands,     np.repeat(pos_brands,     num_neg)])
    all_cats       = np.concatenate([pos_cats,       np.repeat(pos_cats,       num_neg)])
    all_prices     = np.concatenate([pos_prices,     np.repeat(pos_prices,     num_neg)])
    all_hours      = np.concatenate([pos_hours,      np.repeat(pos_hours,      num_neg)])
    all_dayofweeks = np.concatenate([pos_dayofweeks, np.repeat(pos_dayofweeks, num_neg)])
    all_labels     = np.concatenate([np.ones(n), np.zeros(n * num_neg)])

    X = torch.tensor(
        np.stack([all_users, all_items, all_brands, all_cats,
                  all_prices, all_hours, all_dayofweeks], axis=1),
        dtype=torch.long
    )
    y = torch.tensor(all_labels, dtype=torch.float32)
    return TensorDataset(X, y)


def run_pipeline(
    data_dir: str = "data/raw/explicit",
    model_path: str = "models/deepfm_best.pth",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.0001,
    min_interactions: int = 5,
    sample_frac: float = 0.05, # ban đeầu là 0.03
    sample_seed: int = 42,
    num_neg_train: int = 0,   # negative/positive cho train, ban đầu là 4 để train nhanh
    num_neg_eval: int = 10,   # negative/positive cho val/test (chuẩn đánh giá), ban đầu là 99
    skip_preprocess: bool = False,
) -> dict:
    import pandas as pd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processed_train = "data/processed/electronics_train.csv"
    processed_val   = "data/processed/electronics_val.csv"
    processed_test  = "data/processed/electronics_test.csv"

    if skip_preprocess and all(os.path.exists(p) for p in [processed_train, processed_val, processed_test]):
        print("\n[1/6] Bỏ qua — đọc data từ data/processed/")
        train_df = pd.read_csv(processed_train)
        print("\n[2/6] Bỏ qua — data đã được encode và split sẵn")
        val_df   = pd.read_csv(processed_val)
        test_df  = pd.read_csv(processed_test)
        num_users = int(train_df["user_idx"].max()) + 1
        num_items = int(train_df["item_idx"].max()) + 1
        print(f"      Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
        print(f"      Users: {num_users:,}  Items: {num_items:,}")
    else:
        # [1/6] Load & clean
        print("\n[1/6] Loading & cleaning data...")
        df = load_all(data_dir)
        print(f"      Raw rows: {len(df):,}")
        df = validate(df)
        df = clean(df, min_interactions=min_interactions)
        print(f"      After filter: {len(df):,} rows")

        if sample_frac < 1.0:
            users = df["user_id"].unique()
            n_users = max(1, int(len(users) * sample_frac))
            sampled_users = np.random.default_rng(sample_seed).choice(users, size=n_users, replace=False)
            df = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)
            print(f"      After sampling ({sample_frac*100:.0f}%): {len(df):,} rows | {n_users:,} users")

        # [2/6] Encode & split
        print("\n[2/6] Encoding & splitting...")
        df, num_users, num_items = encode(df)
        print(f"      Users: {num_users:,}  Items: {num_items:,}")
        train_df, val_df, test_df = split(df)
        print(f"      Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

        train_df.to_csv(processed_train, index=False)
        val_df.to_csv(processed_val, index=False)
        test_df.to_csv(processed_test, index=False)
        print("      Đã lưu data vào data/processed/")

    # [3/6] Build dataset & dataloader
    print("\n[3/6] Building datasets...")

    # tạo ExplicitDataset để lấy vocab maps
    train_data = ExplicitDataset(train_df)

    # [SỬA] negative sampling nhất quán cho cả 3 tập
    # train: ít negative hơn để train nhanh (num_neg_train=4)
    # val/test: nhiều negative hơn (num_neg_eval=99) để đánh giá ranking chính xác
    print(f"      Đang tạo negative samples cho train ({num_neg_train} neg/pos)...")
    train_dataset = negative_sample(train_df, train_data, num_neg=num_neg_train)

    print(f"      Đang tạo negative samples cho val ({num_neg_eval} neg/pos)...")
    val_dataset   = negative_sample(val_df,   train_data, num_neg=num_neg_eval)

    print(f"      Đang tạo negative samples cho test ({num_neg_eval} neg/pos)...")
    test_dataset  = negative_sample(test_df,  train_data, num_neg=num_neg_eval)

    print(f"      Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"      field_dims: {train_data.field_dims}")

    # [4/6] Train
    print("\n[4/6] Training DeepFM...")
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

    # [5/6] Evaluate
    print("\n[5/6] Evaluating on test set...")
    metrics = evaluate(model, test_loader, device=device)
    print(f"      Test Acc: {metrics['accuracy']:.4f}  AUC: {metrics.get('auc', 'N/A')}")

    config = {
        "field_dims": field_dims.tolist(),
        "num_users": num_users,
        "num_items": num_items,
    }
    save_model(model, model_path, config=config, metadata=metrics)
    print(f"      Model saved → {model_path}")

    # [6/6] Generate results
    print("\n[6/6] Generating results...")
    result_dir = generate_results(
        model, history, metrics,
        test_loader, test_df, train_df,
        train_data.user_map, train_data.item_map, train_data.brand_map,
        device
    )
    print(f"      Results saved → {result_dir}")

    if metrics["accuracy"] < ACC_THRESHOLD:
        print(
            f"\nWARNING: Accuracy {metrics['accuracy']:.4f} below threshold {ACC_THRESHOLD}."
        )

    print("\nDone.")
    return metrics


if __name__ == "__main__":
    run_pipeline()