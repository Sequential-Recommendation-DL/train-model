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
    print("\n [4/4] Saving model and results...")

    config = {
        "num_users": num_users,
        "num_items": num_items,
        "gmf_dim": model._gmf_dim,
        "mlp_dim": model._mlp_input_dim // 2,
        "dropout": model._dropout,
    }
    save_model(
        model, model_path, config=config,
        metadata={"auc_roc": metrics["AUC_ROC"]},
    )

    out_dir = RESULTS_BASE / datetime.now().strftime("%d_%m_%y_%Hh_%Mp")
    _write_plots(out_dir, metrics, history)
    print(f"      Results saved to {out_dir}")


def _write_plots(out_dir, metrics, history):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    (out_dir / "metrics.json").write_text(json.dumps(json_metrics, indent=2))

    raw_scores = metrics.get("_raw_scores", [])
    raw_labels = metrics.get("_raw_labels", [])
    if raw_scores and raw_labels:
        _plot_roc(out_dir, metrics, raw_scores, raw_labels)

    if history:
        _plot_training_history(out_dir, history)


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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [h["loss"] for h in history], marker="o", label="Train Loss")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="s", label="Val Loss")
    ax.set(xlabel="Epoch", ylabel="BCE Loss", title="Loss (overfitting check)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=150)
    plt.close(fig)
