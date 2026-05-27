import sys

import torch
from torch.utils.data import DataLoader

from src.data.load_data import load_all
from src.data.preprocess import clean, encode, split, validate
from src.features.build_features import NCFDataset, build_user_pos, negative_sample
from src.models.neumf import NeuMF
from src.models.predict import evaluate
from src.models.save_load import save_model
from src.models.train import train

HR_THRESHOLD = 0.60
NDCG_THRESHOLD = 0.35


def run_pipeline(
    data_dir: str = "data/raw/explicit",
    model_path: str = "models/neumf_best.pt",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_neg_train: int = 4,
    min_interactions: int = 5,
    max_eval_users: int = 5_000,
) -> dict:  # type: ignore[type-arg]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n[1/6] Loading & cleaning data...")
    df = load_all(data_dir)
    print(f"      Raw rows: {len(df):,}")
    df = validate(df)
    df = clean(df, min_interactions=min_interactions)
    print(f"      After filter: {len(df):,} rows")

    print("\n[2/6] Encoding & splitting...")
    df, num_users, num_items = encode(df)
    print(f"      Users: {num_users:,}  Items: {num_items:,}")
    train_df, val_df, test_df = split(df)
    print(f"      Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    print("\n[3/6] Building training features...")
    user_pos = build_user_pos(train_df)
    train_sampled = negative_sample(train_df, num_items, user_pos, num_neg=num_neg_train)
    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        NCFDataset(train_sampled), batch_size=batch_size, shuffle=True, num_workers=0
    )
    print(f"      Training samples: {len(train_sampled):,}")

    print("\n[4/6] Training NeuMF...")
    model = NeuMF(num_users, num_items)
    model, history = train(
        model,
        train_loader,
        val_df=val_df,
        user_pos=user_pos,
        num_items=num_items,
        epochs=epochs,
        lr=lr,
        device=device,
        max_val_users=max_eval_users,
    )

    print("\n[5/6] Evaluating on test set...")
    metrics = evaluate(
        model, test_df, user_pos, num_items,
        device=device, max_users=max_eval_users,
    )
    print(f"      HR@10={metrics['HR@10']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}")

    print("\n[6/6] Saving model...")
    config = {"num_users": num_users, "num_items": num_items, "gmf_dim": 64, "mlp_dim": 64}
    save_model(model, model_path, config=config, metadata=metrics)

    if metrics["HR@10"] < HR_THRESHOLD or metrics["NDCG@10"] < NDCG_THRESHOLD:
        print(
            f"\nWARNING: Below acceptance threshold "
            f"(HR@10≥{HR_THRESHOLD}, NDCG@10≥{NDCG_THRESHOLD}). "
            "Check data quality or tune hyperparameters."
        )
        sys.exit(1)

    print("\nDone.")
    return metrics


if __name__ == "__main__":
    run_pipeline()
