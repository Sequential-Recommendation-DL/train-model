import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import timer, ensure_dir
from .config import EDA_DIR, TRAIN_PATH, VAL_PATH

plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

SEP = "\u2500" * 50
COLORS = ["#4ECDC4", "#FF6B6B", "#FFE66D", "#95E1D3"]


def _load():
    with timer("  Load processed data"):
        train = pd.read_csv(TRAIN_PATH)
        val = pd.read_csv(VAL_PATH)
        train["_split"] = "train"
        val["_split"] = "val"
        df = pd.concat([train, val], ignore_index=True)
    return train, val, df


def _print_df(name: str, df: pd.DataFrame):
    print(f"\n  {name}: {len(df):,} rows | "
          f"users={df['UserId'].nunique():,} | "
          f"items={df['ItemId'].nunique():,} | "
          f"Label mean={df['Label'].mean():.4f}")


def run():
    ensure_dir(EDA_DIR)

    print(f"\n{SEP}")
    print("  EDA — PROCESSED DATA")
    print(SEP)

    # ── Load ──
    train, val, df = _load()
    print(f"\n  Total: {len(df):,} rows")
    _print_df("Train", train)
    _print_df("Val", val)

    # ── 1. Label distribution ──
    print(f"\n{SEP}")
    print("  1. Label Distribution")
    print(SEP)

    bins = [0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
    labels = ["0.00-0.50", "0.50-1.00", "1.00-1.25", "1.25-1.50", "1.50-1.75", "1.75-2.00"]

    for name, subset in [("Overall", df), ("Train", train), ("Val", val)]:
        binned = pd.cut(subset["Label"], bins=bins, labels=labels).value_counts()
        print(f"\n  {name}:")
        for lbl in labels:
            cnt = binned.get(lbl, 0)
            pct = cnt / len(subset) * 100
            bar = "\u2588" * max(1, int(pct / 3))
            print(f"    {lbl}: {cnt:>8,} ({pct:5.2f}%) {bar}")

    # Chart: Label histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (name, subset) in zip(axes, [("Train", train), ("Val", val)]):
        ax.hist(subset["Label"], bins=40, color=COLORS[0], edgecolor="white", alpha=0.8)
        ax.axvline(subset["Label"].mean(), color=COLORS[1], linestyle="--", label=f"mean={subset['Label'].mean():.3f}")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend()
        ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "label_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved label_distribution.png")

    # ── 2. User & Item stats ──
    print(f"\n{SEP}")
    print("  2. User & Item Statistics")
    print(SEP)

    user_counts = df.groupby("UserId").size()
    item_counts = df.groupby("ItemId").size()

    print(f"\n  Users: {df['UserId'].nunique():,}")
    print(f"  Items: {df['ItemId'].nunique():,}")
    print(f"  Pairs: {len(df):,}")
    sparsity = 1 - len(df) / (df["UserId"].nunique() * df["ItemId"].nunique())
    print(f"  Sparsity: {sparsity:.4%}")
    print(f"\n  Interactions per user:")
    print(f"    mean={user_counts.mean():.2f}  median={user_counts.median():.2f}  "
          f"min={user_counts.min()}  max={user_counts.max()}")
    print(f"  Interactions per item:")
    print(f"    mean={item_counts.mean():.2f}  median={item_counts.median():.2f}  "
          f"min={item_counts.min()}  max={item_counts.max()}")

    # Chart: user/item stats
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(user_counts.values, bins=50, color=COLORS[2], edgecolor="white")
    axes[0].set_xlabel("Interactions per user")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Users")
    axes[0].set_yscale("log")
    axes[1].hist(item_counts.values, bins=50, color=COLORS[3], edgecolor="white")
    axes[1].set_xlabel("Interactions per item")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Items")
    axes[1].set_yscale("log")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "user_item_stats.png", bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved user_item_stats.png")

    # ── 3. Timestamp ──
    print(f"\n{SEP}")
    print("  3. Timestamp")
    print(SEP)

    t_min, t_max = df["Timestamp"].min(), df["Timestamp"].max()
    t_range_s = t_max - t_min
    print(f"  Range: {t_min} .. {t_max} ({t_range_s / 3600:.1f} hours = {t_range_s / 86400:.1f} days)")

    dt_min = pd.to_datetime(t_min, unit="s")
    dt_max = pd.to_datetime(t_max, unit="s")
    print(f"  ({dt_min} .. {dt_max})")

    # ── 4. Compare train vs val ──
    print(f"\n{SEP}")
    print("  4. Train vs Val")
    print(SEP)
    user_overlap = len(set(train["UserId"].unique()) & set(val["UserId"].unique()))
    print(f"\n  {'Metric':<25} {'Train':>12} {'Val':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Rows':<25} {len(train):>12,} {len(val):>12,}")
    print(f"  {'Users':<25} {train['UserId'].nunique():>12,} {val['UserId'].nunique():>12,}")
    print(f"  {'Items':<25} {train['ItemId'].nunique():>12,} {val['ItemId'].nunique():>12,}")
    print(f"  {'Label mean':<25} {train['Label'].mean():>12.4f} {val['Label'].mean():>12.4f}")
    print(f"  {'User overlap':<25} {user_overlap:>12,} ({user_overlap / val['UserId'].nunique() * 100:.1f}% of val)")

    # ── 5. Summary ──
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"\n  Total rows:  {len(df):,}")
    print(f"  Users:       {df['UserId'].nunique():,}")
    print(f"  Items:       {df['ItemId'].nunique():,}")
    print(f"  Sparsity:    {sparsity:.4%}")
    print(f"  Label range: [{df['Label'].min():.4f}, {df['Label'].max():.4f}]")
    print(f"  Label mean:  {df['Label'].mean():.4f}")
    print(f"  Timestamp:   {dt_min} .. {dt_max}")
    print(f"  Charts:      {EDA_DIR}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    run()
