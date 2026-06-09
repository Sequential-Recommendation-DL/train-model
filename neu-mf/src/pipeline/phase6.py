import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.models.save_load import save_model

RESULTS_BASE = Path("results")


def save_results(model, metrics, history, num_users, num_items, model_path="models/neumf_best.pt"):
    print("\n [6/6] Saving model and results...")

    config = {"num_users": num_users, "num_items": num_items, "gmf_dim": 64, "mlp_dim": 64}
    save_model(
        model, model_path, config=config,
        metadata={"hit_rate_at_10": metrics["HR@10"], "ndcg_at_10": metrics["NDCG@10"]},
    )

    out_dir = RESULTS_BASE / datetime.now().strftime("%d_%m_%y_%Hh_%Mp")
    _write_plots(out_dir, metrics, history)
    print(f"      Results saved to {out_dir}")


def _write_plots(out_dir, metrics, history):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    (out_dir / "metrics.json").write_text(json.dumps(json_metrics, indent=2))

    _plot_confusion_matrix(out_dir, metrics)

    raw_scores = metrics.get("_raw_scores", [])
    raw_labels = metrics.get("_raw_labels", [])
    if raw_scores and raw_labels:
        _plot_roc(out_dir, metrics, raw_scores, raw_labels)

    if history:
        _plot_training_history(out_dir, history)


def _plot_confusion_matrix(out_dir, metrics):
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore[attr-defined]
    fig.colorbar(im)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        title="Confusion Matrix (top 10 per user)",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_roc(out_dir, metrics, raw_scores, raw_labels):
    from sklearn.metrics import roc_curve
    probs = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))
    fpr, tpr, _ = roc_curve(raw_labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC_ROC']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_training_history(out_dir, history):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, [h["loss"] for h in history], marker="o", label="Train Loss")
    ax1.plot(epochs, [h.get("val_loss", 0) for h in history], marker="s", label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss per Batch", title="Loss (overfitting check)")
    ax1.legend()
    ax2.plot(epochs, [h.get("HR@10", 0) for h in history], marker="o", label="Hit Rate at 10")
    ax2.plot(epochs, [h.get("NDCG@10", 0) for h in history], marker="s", label="NDCG at 10")
    ax2.set(xlabel="Epoch", ylabel="Score", title="Validation Metrics")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=150)
    plt.close(fig)
