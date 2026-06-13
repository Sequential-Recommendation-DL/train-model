import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import EDA_DIR
from .utils import ensure_dir


def normalize_comparison():
    ensure_dir(EDA_DIR)

    x = np.linspace(0, 20, 500)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    tanh = np.tanh(x / 5.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, sigmoid, label=r"$\sigma(x)$", linewidth=2, color="#FF6B6B")
    ax.plot(x, tanh, label=r"$\tanh(x / 5)$", linewidth=2, color="#4ECDC4")

    behaviors = {"pv": 1, "fav": 2, "cart": 3, "buy": 4}
    colors = {"pv": "#95E1D3", "fav": "#FFE66D", "cart": "#FF6B6B", "buy": "#4ECDC4"}
    for name, score in behaviors.items():
        sig_v = 1.0 / (1.0 + np.e**-score)
        tan_v = np.tanh(score / 5.0)
        ax.axvline(score, color=colors[name], linestyle=":", alpha=0.5, linewidth=1)
        ax.plot(score, sig_v, "o", color="#FF6B6B", markersize=6)
        ax.plot(score, tan_v, "o", color="#4ECDC4", markersize=6)
        ax.annotate(f"  {name}({score})", (score, 0.75), fontsize=9, rotation=90,
                    va="center", ha="left", alpha=0.7)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.fill_between(x, sigmoid, tanh, alpha=0.05, color="gray")

    ax.set_xlabel("Raw Label (sum of behavior scores)", fontsize=11)
    ax.set_ylabel("Normalized Label", fontsize=11)
    ax.set_title("So sánh: Sigmoid vs Tanh(x/5)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.2)

    ax.annotate("Label >=4 -> gần như bằng\n(khó phân biệt)",
                xy=(4, 0.98), xytext=(6.5, 0.85),
                arrowprops=dict(arrowstyle="->", color="#FF6B6B", alpha=0.5),
                fontsize=9, color="#FF6B6B")
    ax.annotate("Label=1..4 giãn đều\n(dễ phân biệt hơn)",
                xy=(4, 0.664), xytext=(6.5, 0.4),
                arrowprops=dict(arrowstyle="->", color="#4ECDC4", alpha=0.5),
                fontsize=9, color="#4ECDC4")

    plt.tight_layout()
    path = EDA_DIR / "normalize_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    normalize_comparison()
