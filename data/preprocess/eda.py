import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from .utils import timer, ensure_dir
from .config import EDA_DIR, TRAIN_PATH, VAL_PATH

plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

SEP = "\u2500" * 50
COLORS = ["#4ECDC4", "#FF6B6B", "#FFE66D", "#95E1D3", "#A8E6CF"]


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
    print("  4. Train vs Val Split Quality")
    print(SEP)

    # ── 4a. Basic metrics ──
    user_overlap = len(set(train["UserId"].unique()) & set(val["UserId"].unique()))
    train_items_set = set(train["ItemId"].unique())
    val_items_set = set(val["ItemId"].unique())
    item_overlap = len(train_items_set & val_items_set)
    val_items_unseen = len(val_items_set - train_items_set)

    print(f"\n  {'Metric':<30} {'Train':>12} {'Val':>12} {'Diff':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Rows':<30} {len(train):>12,} {len(val):>12,} {'—':>10}")
    print(f"  {'Users':<30} {train['UserId'].nunique():>12,} {val['UserId'].nunique():>12,} {'—':>10}")
    print(f"  {'Items':<30} {len(train_items_set):>12,} {len(val_items_set):>12,} {'—':>10}")
    print(f"  {'Label mean':<30} {train['Label'].mean():>12.4f} {val['Label'].mean():>12.4f} {abs(train['Label'].mean() - val['Label'].mean()):>10.4f}")
    print(f"  {'Label std':<30} {train['Label'].std():>12.4f} {val['Label'].std():>12.4f} {abs(train['Label'].std() - val['Label'].std()):>10.4f}")
    print(f"  {'Label p50 (median)':<30} {train['Label'].median():>12.4f} {val['Label'].median():>12.4f} {abs(train['Label'].median() - val['Label'].median()):>10.4f}")
    print(f"  {'Label p90':<30} {train['Label'].quantile(0.9):>12.4f} {val['Label'].quantile(0.9):>12.4f} {abs(train['Label'].quantile(0.9) - val['Label'].quantile(0.9)):>10.4f}")
    ks_stat, ks_pval = ks_2samp(train["Label"], val["Label"])
    print(f"  {'KS test stat (p)':<30} {'':>12} {ks_stat:.4f} ({ks_pval:.4f}) {'':>10}")
    print(f"  {'User overlap':<30} {'':>12} {user_overlap:,} ({user_overlap / val['UserId'].nunique() * 100:.1f}% of val)")
    print(f"  {'Item overlap':<30} {'':>12} {item_overlap:,} ({item_overlap / len(val_items_set) * 100:.1f}% of val)")
    print(f"  {'Val items unseen':<30} {'':>12} {val_items_unseen:,} ({val_items_unseen / len(val_items_set) * 100:.1f}%)")

    # ── 4b. User behavior profile ──
    print(f"\n  User profile (by # interactions):")
    def user_profile(df):
        counts = df.groupby("UserId").size()
        return pd.Series({
            "1-timer": (counts == 1).sum(),
            "light (2-3)": ((counts >= 2) & (counts <= 3)).sum(),
            "medium (4-5)": ((counts >= 4) & (counts <= 5)).sum(),
            "heavy (>5)": (counts > 5).sum(),
        })
    train_profile = user_profile(train)
    val_profile = user_profile(val)
    print(f"  {'Category':<20} {'Train':>10} {'Val':>10} {'Diff':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for cat in train_profile.index:
        tp = train_profile[cat] / len(train["UserId"].unique()) * 100
        vp = val_profile[cat] / len(val["UserId"].unique()) * 100
        print(f"  {cat:<20} {tp:>9.1f}% {vp:>9.1f}% {abs(tp - vp):>9.1f}%")

    # ── 4c. Item popularity profile ──
    print(f"\n  Item profile (by # users in train):")
    train_item_pop = train.groupby("ItemId").size()
    def item_pop_bucket(n):
        if n == 1: return "1 user"
        elif n <= 5: return "2-5 users"
        elif n <= 20: return "6-20 users"
        else: return ">20 users"
    item_buckets = train_item_pop.map(item_pop_bucket).value_counts()
    for bucket in ["1 user", "2-5 users", "6-20 users", ">20 users"]:
        cnt = item_buckets.get(bucket, 0)
        print(f"    {bucket:<15}: {cnt:>6,} items ({cnt / len(train_item_pop) * 100:.1f}%)")

    # ── 4d. Temporal distribution ──
    print(f"\n  Temporal distribution by day:")
    train_daily = train.assign(_day=pd.to_datetime(train["Timestamp"], unit="s").dt.date).groupby("_day").size()
    val_daily = val.assign(_day=pd.to_datetime(val["Timestamp"], unit="s").dt.date).groupby("_day").size()
    all_days = sorted(set(list(train_daily.index) + list(val_daily.index)))
    print(f"  {'Date':<15} {'Train':>8} {'Val':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}")
    for d in all_days:
        print(f"  {str(d):<15} {train_daily.get(d, 0):>8,} {val_daily.get(d, 0):>8,}")

    # Chart: Overlay label distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train["Label"], bins=40, alpha=0.6, color=COLORS[0], label=f"Train (mean={train['Label'].mean():.3f})")
    ax.hist(val["Label"], bins=40, alpha=0.6, color=COLORS[1], label=f"Val (mean={val['Label'].mean():.3f})")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution: Train vs Val")
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(EDA_DIR / "label_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved label_comparison.png")

    # Chart: User profile comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = list(train_profile.index)
    x = np.arange(len(cats))
    w = 0.35
    train_pct = [train_profile[c] / len(train["UserId"].unique()) * 100 for c in cats]
    val_pct = [val_profile[c] / len(val["UserId"].unique()) * 100 for c in cats]
    ax.bar(x - w/2, train_pct, w, color=COLORS[0], label="Train")
    ax.bar(x + w/2, val_pct, w, color=COLORS[1], label="Val")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("% of users")
    ax.set_title("User Profile: Train vs Val")
    ax.legend()
    fig.savefig(EDA_DIR / "user_profile_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"  -> Saved user_profile_comparison.png")

    # Chart: Temporal distribution
    fig, ax = plt.subplots(figsize=(12, 4))
    tmp = pd.concat([train.assign(_s="train"), val.assign(_s="val")])
    tmp["_day"] = pd.to_datetime(tmp["Timestamp"], unit="s").dt.date
    pivot = tmp.groupby(["_day", "_s"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", ax=ax, color=[COLORS[0], COLORS[1]])
    ax.set_xlabel("Date")
    ax.set_ylabel("Rows")
    ax.set_title("Samples per Day: Train vs Val")
    ax.legend()
    fig.savefig(EDA_DIR / "temporal_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"  -> Saved temporal_comparison.png")

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
