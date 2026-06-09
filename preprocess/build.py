import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import timer, ensure_dir
from config import (
    RAW_DATA,
    PROCESSED_DIR,
    COLUMNS,
    DTYPES,
    TRAIN_PATH,
    VAL_PATH,
    METADATA_PATH,
    TIMESTAMP_RANGE,
    TRAIN_RATIO,
    RANDOM_SEED,
)

SEP = "\u2500" * 50


def run(n_rows: int | None = None):
    ensure_dir(PROCESSED_DIR)

    print(f"\n{SEP}")
    print("  BUILD PIPELINE")
    print(SEP)

    # ── 1. Load ──
    with timer("1. Load"):
        df = pd.read_csv(RAW_DATA, names=COLUMNS, dtype=DTYPES)
    n_loaded = len(df)
    print(f"     Rows: {n_loaded:,}")
    print(f"     Columns: {list(df.columns)}")

    # ── 2. Clean ──
    with timer("2. Clean"):
        n_before = len(df)
        n_dup = df.duplicated().sum()
        df = df.drop_duplicates()
        t_min, t_max = TIMESTAMP_RANGE
        n_outside = (~df["timestamp"].between(t_min, t_max)).sum()
        df = df[df["timestamp"].between(t_min, t_max)]
    print(f"     After dedup:        {n_before - n_dup:,} / {n_before:,}")
    print(f"     After timestamp:    {len(df):,} / {n_before:,} (range {t_min}..{t_max})")

    # ── 3. Score ──
    with timer("3. Score behavior"):
        score_map = {"pv": 1, "fav": 2, "cart": 3, "buy": 4}
        df["label"] = df["behavior"].map(score_map).astype("int8")
    dist = df["label"].value_counts().sort_index()
    print(f"     Score distribution ({len(df):,} rows):")
    for beh, score in score_map.items():
        cnt = dist.get(score, 0)
        print(f"       {beh:5} ({score}) : {cnt:>10,} ({cnt / len(df) * 100:5.2f}%)")

    # ── 4. Groupby ──
    with timer("4. Group by (user, item)"):
        n_before_gb = len(df)
        g = df.groupby(["user_id", "item_id"], as_index=False)
        df = g.agg(Timestamp=("timestamp", "max"), Label=("label", "sum"))
        df.columns = ["UserId", "ItemId", "Timestamp", "Label"]
    compression = (1 - len(df) / n_before_gb) * 100
    n_users = df["UserId"].nunique()
    n_items = df["ItemId"].nunique()
    sparsity = 1 - len(df) / (n_users * n_items) if n_users * n_items > 0 else 0
    print(f"     Users: {n_users:,}  Items: {n_items:,}  Sparsity: {sparsity:.4%}")
    print(f"     {n_before_gb:,} rows \u2192 {len(df):,} rows ({compression:.1f}% reduction)")

    # ── 4b. Take top N by timestamp ──
    if n_rows is not None and n_rows < len(df):
        with timer("4b. Take top N by timestamp"):
            df = df.sort_values("Timestamp", ascending=False).head(n_rows).reset_index(drop=True)
        print(f"     Took {n_rows:,} most recent (user, item) pairs")
        n_users = df["UserId"].nunique()
        n_items = df["ItemId"].nunique()

    # Label distribution
    print(f"\n     Label distribution (top 10):")
    label_dist = df["Label"].value_counts().sort_index().head(10)
    for label, cnt in label_dist.items():
        pct = cnt / len(df) * 100
        bar = "\u2588" * max(1, int(pct / 2))
        print(f"       {label:>3}: {cnt:>8,} ({pct:5.2f}%) {bar}")

    # ── 5. Split ──
    with timer("5. Split"):
        train, val = train_test_split(
            df,
            test_size=1 - TRAIN_RATIO,
            random_state=RANDOM_SEED,
        )
    total = len(train) + len(val)
    print(f"     Train: {len(train):,} ({len(train) / total * 100:.1f}%)")
    print(f"     Val:   {len(val):,} ({len(val) / total * 100:.1f}%)")
    print(f"     Label mean: train={train['Label'].mean():.2f}  val={val['Label'].mean():.2f}")

    # ── 6. Save ──
    with timer("6. Save"):
        train.to_csv(TRAIN_PATH, index=False)
        val.to_csv(VAL_PATH, index=False)
    print(f"     Train: {TRAIN_PATH} ({os.path.getsize(TRAIN_PATH) / 1024 / 1024:.1f} MB)")
    print(f"     Val:   {VAL_PATH} ({os.path.getsize(VAL_PATH) / 1024 / 1024:.1f} MB)")

    # ── 7. Metadata ──
    with timer("7. Metadata"):
        metadata = {
            "n_rows_take": n_rows,
            "n_rows_raw": n_loaded,
            "n_rows_clean": total,
            "n_train": len(train),
            "n_val": len(val),
            "n_users": n_users,
            "n_items": n_items,
            "score_distribution": {str(k): int(v) for k, v in df["Label"].value_counts().sort_index().items()},
            "timestamp_range": list(TIMESTAMP_RANGE),
            "train_ratio": TRAIN_RATIO,
            "random_seed": RANDOM_SEED,
            "columns": ["UserId", "ItemId", "Timestamp", "Label"],
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
    print(f"     {METADATA_PATH}")

    print(f"\n{SEP}")
    print("  DONE")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None, help="250000 or 500000")
    args = parser.parse_args()
    run(n_rows=args.rows)
