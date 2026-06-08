import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
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
from src.pipeline.phase1 import loading_and_clean_data
from src.pipeline.phase2 import encodingAndSplitting
from src.utils.check_gpu import check_gpu
from src.utils.min_interactions import MinInteractions
from src.utils.load_cache_data import loadCacheData

HR_THRESHOLD = 0.60
NDCG_THRESHOLD = 0.35
RESULTS_BASE = Path("results")
PROCESSED_DIR = Path("data/processed")

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
        title="Confusion Matrix (top-10 per user)",
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


'''
def run_pipeline(
    data_dir: str = "data/raw/explicit",
    model_path: str = "models/neumf_best.pt",
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    num_neg_train: int = 15,
    min_interactions: int = 5,
    max_eval_users: int = 2_000,
    max_users: int | None = 250_000,  # total cap; split equally across categories to prevent bias
) -> dict:  # type: ignore[type-arg]
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
        patience=5,
        max_val_users=max_eval_users,
    )
    train_finished_at = datetime.now()

    print("\n[5/6] Evaluating on test set...")
    metrics = evaluate_full(
        model, test_df, user_pos, num_items,
        device=device, max_users=max_eval_users,
    )
    n_hits = metrics["n_hits_at_k"]
    n_total = metrics["n_eval_users"]
    ndcg_sum = round(metrics["NDCG@10"] * n_total, 1)
    cm = metrics["confusion_matrix"]
    tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
    print(
        f"      HR@10   = hits/users = {n_hits}/{n_total} = {metrics['HR@10']:.4f}\n"
        f"               [positive item in top-10 of 100 candidates → 1, else 0]\n"
        f"      NDCG@10 = Σ(1/log₂(rank+1))/users = {ndcg_sum}/{n_total} = {metrics['NDCG@10']:.4f}\n"
        f"               [rank of positive; gain = 1/log₂(rank+1) if rank≤10, else 0]\n"
        f"      AUC-ROC = {metrics['AUC_ROC']:.4f}  [P(score_pos > score_neg) over all pos/neg pairs]\n"
        f"      Precision = TP/(TP+FP) = {tp}/{tp+fp} = {metrics['Precision']:.4f}  [top-10 per user]\n"
        f"      Recall    = TP/(TP+FN) = {tp}/{tp+fn} = {metrics['Recall']:.4f}  [== HR@10]\n"
        f"      F1        = 2·P·R/(P+R) = {metrics['F1']:.4f}\n"
        f"      Confusion : TN={cm[0][0]}  FP={fp}  FN={fn}  TP={tp}  [ranking-based, opt_thr={metrics.get('optimal_threshold', 0):.3f}]"
    )

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
'''

def run_pipeline():
    device = check_gpu()
    minInteractions = MinInteractions
    cache_key = f"mi{minInteractions.value}"
    cached = loadCacheData(cache_key)
    if cached:
        train_df, val_df, test_df, num_users, num_items = cached
        print(f"\n [2/6] Loaded processed data from cache (key={cache_key})")
        print(f" - Users: {num_users:,}  Items: {num_items:,}")
        print(f" - Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    else:
        df = loading_and_clean_data(minInteractions)
        train_df, val_df, test_df, num_users, num_items = encodingAndSplitting(df, cache_key)


if __name__ == "__main__":
    run_pipeline()
