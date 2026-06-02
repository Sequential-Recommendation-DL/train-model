import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.load_data import load_all
from src.data.preprocess import clean, encode, split, validate
from src.features.build_features import NCFDataset, build_user_pos, negative_sample
from src.models.neumf import NeuMF
from src.models.predict import evaluate_full
from src.models.save_load import save_model
from src.models.train import train

HR_THRESHOLD = 0.60
NDCG_THRESHOLD = 0.35
RESULTS_BASE = Path("results")


def _save_results(out_dir: Path, metrics: dict, history: list[dict]) -> None:  # type: ignore[type-arg]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_scores = metrics.pop("_raw_scores", [])
    raw_labels = metrics.pop("_raw_labels", [])

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore[attr-defined]
    fig.colorbar(im)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        title="Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    if raw_scores and raw_labels:
        from sklearn.metrics import roc_curve
        probs = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))
        fpr, tpr, _ = roc_curve(raw_labels, probs)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC={metrics['AUC_ROC']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close(fig)

    # Training history — train vs val loss for overfitting detection
    if history:
        epochs = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, [h["loss"] for h in history], marker="o", label="Train Loss")
        ax1.plot(epochs, [h.get("val_loss", 0) for h in history], marker="s", label="Val Loss")
        ax1.set(xlabel="Epoch", ylabel="Avg Loss / Batch", title="Loss Curve (overfitting check)")
        ax1.legend()
        ax2.plot(epochs, [h.get("HR@10", 0) for h in history], marker="o", label="HR@10")
        ax2.plot(epochs, [h.get("NDCG@10", 0) for h in history], marker="s", label="NDCG@10")
        ax2.set(xlabel="Epoch", ylabel="Score", title="Validation Metrics")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_history.png", dpi=150)
        plt.close(fig)

    print(f"      Results saved → {out_dir}")


def run_pipeline(
    data_dir: str = "data/raw/explicit",
    model_path: str = "models/neumf_best.pt",
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    num_neg_train: int = 2,
    min_interactions: int = 5,
    max_eval_users: int = 2_000,
    max_users: int | None = 100_000,  # cap dataset so training finishes in ~1h on RTX 3050 6GB
) -> dict:  # type: ignore[type-arg]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"Device: {device} ({gpu.name}, {gpu.total_memory // 1024**2} MB VRAM)")
    else:
        print(f"Device: {device}")

    print("\n[1/6] Loading & cleaning data...")
    df = load_all(data_dir)
    print(f"      Raw rows: {len(df):,}")
    df = validate(df)
    df = clean(df, min_interactions=min_interactions)
    print(f"      After filter: {len(df):,} rows")

    if max_users is not None:
        unique_users = np.asarray(df["user_id"].unique())
        if len(unique_users) > max_users:
            keep = np.random.default_rng(42).choice(unique_users, size=max_users, replace=False)
            df = df[df["user_id"].isin(keep.tolist())].reset_index(drop=True)  # type: ignore[assignment]
            print(f"      Capped to {max_users:,} users: {len(df):,} rows remaining")

    print("\n[2/6] Encoding & splitting...")
    df, num_users, num_items = encode(df)  # type: ignore[arg-type]
    print(f"      Users: {num_users:,}  Items: {num_items:,}")
    train_df, val_df, test_df = split(df)
    print(f"      Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    print("\n[3/6] Building training features...")
    user_pos = build_user_pos(train_df)
    train_sampled = negative_sample(train_df, num_items, user_pos, num_neg=num_neg_train)
    pin = device == "cuda"
    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        NCFDataset(train_sampled), batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=pin, persistent_workers=True,
    )
    print(f"      Training samples: {len(train_sampled):,}  Batches/epoch: {len(train_loader):,}")

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
    train_finished_at = datetime.now()

    print("\n[5/6] Evaluating on test set...")
    metrics = evaluate_full(
        model, test_df, user_pos, num_items,
        device=device, max_users=max_eval_users,
    )
    print(
        f"      HR@10={metrics['HR@10']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}\n"
        f"      AUC-ROC={metrics['AUC_ROC']:.4f}  Precision={metrics['Precision']:.4f}"
        f"  Recall={metrics['Recall']:.4f}  F1={metrics['F1']:.4f}"
    )
    cm = metrics["confusion_matrix"]
    print(f"      Confusion Matrix: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    out_dir = RESULTS_BASE / train_finished_at.strftime("%d_%m_%y_%Hh_%Mp")
    _save_results(out_dir, metrics, history)

    print("\n[6/6] Saving model...")
    config = {"num_users": num_users, "num_items": num_items, "gmf_dim": 64, "mlp_dim": 64}
    save_model(model, model_path, config=config, metadata={"HR@10": metrics["HR@10"], "NDCG@10": metrics["NDCG@10"]})

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
