import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.load_data import load_all
from src.data.preprocess import clean, encode, split, validate
from src.models.deepfm import DeepFM, ExplicitDataset
from src.models.predict import evaluate
from src.models.save_load import save_model
from src.models.train import train

ACC_THRESHOLD = 0.75


def run_pipeline(
    data_dir: str = "data/raw/explicit",
    model_path: str = "models/deepfm_best.pth",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    min_interactions: int = 5,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # [1/5] Load & clean
    print("\n[1/5] Loading & cleaning data...")
    df = load_all(data_dir)
    print(f"      Raw rows: {len(df):,}")
    df = validate(df)
    df = clean(df, min_interactions=min_interactions)
    print(f"      After filter: {len(df):,} rows")

    # [2/5] Encode & split
    print("\n[2/5] Encoding & splitting...")
    df, num_users, num_items = encode(df)
    print(f"      Users: {num_users:,}  Items: {num_items:,}")
    train_df, val_df, test_df = split(df)
    print(f"      Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # lưu processed data
    train_df.to_csv("data/processed/electronics_train.csv", index=False)
    val_df.to_csv("data/processed/electronics_val.csv", index=False)
    test_df.to_csv("data/processed/electronics_test.csv", index=False)

    # [3/5] Build dataset & dataloader
    print("\n[3/5] Building datasets...")
    train_data = ExplicitDataset(train_df)
    val_data   = ExplicitDataset(val_df,
                                  user_map=train_data.user_map,
                                  item_map=train_data.item_map,
                                  time_map=train_data.time_map)
    test_data  = ExplicitDataset(test_df,
                                  user_map=train_data.user_map,
                                  item_map=train_data.item_map,
                                  time_map=train_data.time_map)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, num_workers=0)
    print(f"      Train samples: {len(train_data):,}")

    # [4/5] Train
    print("\n[4/5] Training DeepFM...")
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

    # [5/5] Evaluate & save
    print("\n[5/5] Evaluating on test set...")
    metrics = evaluate(model, test_loader, device=device)
    print(f"      Test Acc: {metrics['accuracy']:.4f}")

    config = {
        "field_dims": field_dims.tolist(),
        "num_users": num_users,
        "num_items": num_items,
    }
    save_model(model, model_path, config=config, metadata=metrics)
    print(f"      Model saved → {model_path}")

    if metrics["accuracy"] < ACC_THRESHOLD:
        print(
            f"\nWARNING: Accuracy {metrics['accuracy']:.4f} below threshold {ACC_THRESHOLD}. "
            "Check data quality or tune hyperparameters."
        )
        sys.exit(1)

    print("\nDone.")
    return metrics


if __name__ == "__main__":
    run_pipeline()