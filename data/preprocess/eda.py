import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import timer, ensure_dir
from .config import RAW_DATA, COLUMNS, DTYPES, EDA_DIR, RANDOM_SEED

plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

SEP = "=" * 60


def print_section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def run(n_rows: int | None = None):
    ensure_dir(EDA_DIR)

    with timer("Load data"):
        df = pd.read_csv(RAW_DATA, names=COLUMNS, dtype=DTYPES)

    n_total = len(df)
    print(f"Total rows: {n_total:,}")

    if n_rows is not None and n_rows < n_total:
        df = df.sample(n=n_rows, random_state=RANDOM_SEED)
        print(f"Sampled: {n_rows:,} / {n_total:,} ({n_rows / n_total * 100:.2f}%)\n")

    # ── 1. Missing values & duplicates ──
    print_section("1. Missing Values & Duplicates")
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing.to_string())
    n_dup = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_dup:,} ({n_dup / n_total * 100:.2f}%)")

    # ── 2. Behavior distribution ──
    print_section("2. Behavior Distribution")
    beh_counts = df["behavior"].value_counts()
    beh_pct = df["behavior"].value_counts(normalize=True)
    beh_summary = pd.DataFrame({"count": beh_counts, "percent": beh_pct * 100})
    print(beh_summary.to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2)
    colors = ["#4ECDC4", "#FF6B6B", "#FFE66D", "#95E1D3"]
    ax1.bar(beh_counts.index, beh_counts.values, color=colors)
    ax1.set_title("Behavior Counts")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)

    wedges, texts, autotexts = ax2.pie(
        beh_counts.values,
        labels=beh_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax2.set_title("Behavior Distribution")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "behavior_distribution.png", bbox_inches="tight")
    plt.close()
    print("  → Saved behavior_distribution.png")

    # ── 3. Basic counts ──
    print_section("3. Unique Counts")
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_cats = df["category_id"].nunique()
    print(f"Users:     {n_users:>12,}")
    print(f"Items:     {n_items:>12,}")
    print(f"Categories:{n_cats:>12,}")
    sparsity = 1 - n_total / (n_users * n_items)
    print(f"Sparsity:  {sparsity:.4%}")

    # ── 4. User statistics ──
    print_section("4. User Interaction Statistics")
    user_counts = df["user_id"].value_counts()
    print(user_counts.describe().to_string())
    print(f"\nUsers with ≥5 interactions: {(user_counts >= 5).sum():,} ({(user_counts >= 5).sum() / n_users * 100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(user_counts.values, bins=50, color="#4ECDC4", edgecolor="white")
    axes[0].set_xlabel("Interactions per user")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Interactions per User")
    axes[0].set_yscale("log")

    axes[1].hist(
        user_counts.values,
        bins=50,
        color="#4ECDC4",
        edgecolor="white",
        cumulative=-1,
    )
    axes[1].set_xlabel("Interactions per user")
    axes[1].set_ylabel("Users with ≥ X interactions")
    axes[1].set_title("Complementary CDF (log-log)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "user_stats.png", bbox_inches="tight")
    plt.close()
    print("  → Saved user_stats.png")

    # ── 4b. Buyer statistics ──
    buyer_mask = df["behavior"] == "buy"
    buyers = df.loc[buyer_mask, "user_id"].unique()
    print(f"Users who bought: {len(buyers):,} ({len(buyers) / n_users * 1:.1f}%)")

    # ── 5. Item statistics ──
    print_section("5. Item Interaction Statistics")
    item_counts = df["item_id"].value_counts()
    print(item_counts.describe().to_string())
    print(f"\nItems with ≥5 interactions: {(item_counts >= 5).sum():,} ({(item_counts >= 5).sum() / n_items * 100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(item_counts.values, bins=50, color="#FF6B6B", edgecolor="white")
    axes[0].set_xlabel("Interactions per item")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Interactions per Item")
    axes[0].set_yscale("log")

    axes[1].hist(
        item_counts.values,
        bins=50,
        color="#FF6B6B",
        edgecolor="white",
        cumulative=-1,
    )
    axes[1].set_xlabel("Interactions per item")
    axes[1].set_ylabel("Items with ≥ X interactions")
    axes[1].set_title("Complementary CDF (log-log)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "item_stats.png", bbox_inches="tight")
    plt.close()
    print("  → Saved item_stats.png")

    # ── 6. Category statistics ──
    print_section("6. Category Statistics")
    cat_counts = df["category_id"].value_counts()
    print(cat_counts.describe().to_string())
    print(f"\nTop 10 categories by interaction count:")
    top10_cat = cat_counts.head(10)
    for cid, cnt in top10_cat.items():
        print(f"  cat {cid:>8}: {cnt:>10,}")

    # ── 7. Time analysis ──
    print_section("7. Time Analysis")
    with timer("  Parse timestamps"):
        dt = pd.to_datetime(df["timestamp"], unit="s")
    df["hour"] = dt.dt.hour.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")
    df["date"] = dt.dt.date

    print("Daily interaction counts:")
    daily_counts = df["date"].value_counts().sort_index()
    for d, cnt in daily_counts.items():
        print(f"  {d}: {cnt:>10,}")

    print("\nHourly distribution:")
    hourly = df["hour"].value_counts().sort_index()
    for h, cnt in hourly.items():
        bar = "█" * max(1, cnt // max(hourly.values) * 40)
        print(f"  {h:02d}:00  {bar}  {cnt:,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(hourly.index, hourly.values, color="#95E1D3", edgecolor="white")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Interactions")
    axes[0].set_title("Activity by Hour")
    axes[0].set_xticks(range(24))
    axes[0].tick_params(axis="x", rotation=45)

    weekday_counts = (
        df["day_of_week"].value_counts().sort_index()
    )
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(
        weekday_counts.index, weekday_counts.values, color="#FFE66D", edgecolor="white"
    )
    axes[1].set_xlabel("Day of week")
    axes[1].set_ylabel("Interactions")
    axes[1].set_title("Activity by Day of Week")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_labels)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "time_series.png", bbox_inches="tight")
    plt.close()
    print("  → Saved time_series.png")

    # ── 8. Conversion funnel ──
    print_section("8. Conversion Funnel (pv → fav → cart → buy)")
    funnel_counts = {}
    for beh in ["pv", "fav", "cart", "buy"]:
        funnel_counts[beh] = df[df["behavior"] == beh]["user_id"].nunique()

    funnel_order = ["pv", "fav", "cart", "buy"]
    print(f"{'Behavior':<10} {'Unique Users':>15} {'Conversion':>15}")
    for i, beh in enumerate(funnel_order):
        cnt = funnel_counts[beh]
        if i == 0:
            conv = 100.0
        else:
            conv = cnt / funnel_counts[funnel_order[0]] * 100
        print(f"  {beh:<10} {cnt:>15,} {conv:>14.2f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(funnel_order))
    ax.bar(x_pos, [funnel_counts[b] for b in funnel_order], color=colors[:4], edgecolor="white", width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(funnel_order)
    ax.set_ylabel("Unique Users")
    ax.set_title("Conversion Funnel")
    for i, (b, v) in enumerate(
        [(b, funnel_counts[b]) for b in funnel_order]
    ):
        ax.text(i, v + max(funnel_counts.values()) * 0.01, f"{v:,}", ha="center")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "funnel.png", bbox_inches="tight")
    plt.close()
    print("  → Saved funnel.png")

    # ── 9. Behavior transition ──
    print_section("9. Per-User Behavior Composition")
    user_beh_matrix = df.pivot_table(
        index="user_id",
        columns="behavior",
        aggfunc="size",
        fill_value=0,
    )
    print("Average behavior counts per user:")
    print(user_beh_matrix.mean().round(2).to_string())
    print("\nMedian behavior counts per user:")
    print(user_beh_matrix.median().round(2).to_string())

    # ── 10. Summary ──
    print(f"\n{SEP}")
    print("EDA Complete")
    print(f"  Charts saved to: {EDA_DIR}")
    print(f"  Total rows: {n_total:,}")
    print(f"  Users: {n_users:,} | Items: {n_items:,} | Categories: {n_cats:,}")
    print(f"  Sparsity: {sparsity:.4%}")
    print(f"  Behavior: {beh_counts.to_dict()}")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None, help="sample N rows from raw data")
    args = parser.parse_args()
    run(n_rows=args.rows)
