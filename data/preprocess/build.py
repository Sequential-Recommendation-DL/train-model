import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import timer, ensure_dir
from .config import (
    RAW_DATA,
    PROCESSED_DIR,
    COLUMNS,
    DTYPES,
    TRAIN_PATH,
    VAL_PATH,
    METADATA_PATH,
    TIMESTAMP_RANGE,
    MIN_ITEM_INTERACTIONS,
    HOUR_BIN_SIZE,
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
        # g = df.groupby(["user_id", "item_id"], as_index=False)
        # df = g.agg(Timestamp=("timestamp", "max"), Label=("label", "sum"))
        # df.columns = ["UserId", "ItemId", "Timestamp", "Label"]
        g = df.groupby(["user_id", "item_id"], as_index=False)
        df = g.agg(
            Timestamp=("timestamp", "max"),
            Label=("label", "sum"),
            CategoryId=("category_id", "first"),
        )
        df.columns = ["UserId", "ItemId", "Timestamp", "Label", "CategoryId"]
    compression = (1 - len(df) / n_before_gb) * 100
    n_users = df["UserId"].nunique()
    n_items = df["ItemId"].nunique()
    sparsity = 1 - len(df) / (n_users * n_items) if n_users * n_items > 0 else 0
    print(f"     Users: {n_users:,}  Items: {n_items:,}  Sparsity: {sparsity:.4%}")
    print(f"     {n_before_gb:,} rows \u2192 {len(df):,} rows ({compression:.1f}% reduction)")

    # ── 4b. Stratified sample by hour ──
    if n_rows is not None and n_rows < len(df):
        with timer("4b. Stratified sample by hour"):
            df["_hour_bin"] = df["Timestamp"] // (HOUR_BIN_SIZE * 3600)
            freq = df["_hour_bin"].value_counts(normalize=True)
            samples_per_bin = (freq * n_rows).round().astype(int)
            parts = []
            actual = 0
            for bin_val, n_take in samples_per_bin.items():
                bin_df = df[df["_hour_bin"] == bin_val]
                n_take = min(n_take, len(bin_df))
                if n_take > 0:
                    parts.append(bin_df.sample(n=n_take, random_state=RANDOM_SEED))
                    actual += n_take
            df = pd.concat(parts, ignore_index=True)
            df = df.drop(columns=["_hour_bin"])
        n_hours = int((df["Timestamp"].max() - df["Timestamp"].min()) / 3600)
        print(f"     Sampled {actual:,} rows across time buckets")
        print(f"     Timestamp span: ~{n_hours} hours")
        n_users = df["UserId"].nunique()
        n_items = df["ItemId"].nunique()

    # ── 4c. Normalize Label to (0, 2) ──
    with timer("4c. Normalize Label to (0, 2)"):
        raw_min = df["Label"].min()
        raw_max = df["Label"].max()
        df["Label"] = 2.0 * np.tanh(df["Label"].to_numpy(dtype=np.float64) / 5.0)
        df["Label"] = df["Label"].astype(np.float32)
    print(f"     Raw range: [{raw_min}, {raw_max}] \u2192 (0, 2) via 2*tanh(x/5)")

    # Label distribution
    print(f"\n     Label distribution (binned):")
    bins = [0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
    labels_bin = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    binned = pd.cut(df["Label"], bins=bins, labels=labels_bin).value_counts()
    for lbl, cnt in binned.items():
        print(f"       {lbl:>9}: {cnt:>8,} ({cnt / len(df) * 100:5.2f}%)")

    # ── 5. Split by user ──
    with timer("5. Split by user"):
        unique_users = pd.Series(df["UserId"].unique())
        train_users, val_users = train_test_split(
            unique_users,
            test_size=1 - TRAIN_RATIO,
            random_state=RANDOM_SEED,
        )
        train = df[df["UserId"].isin(train_users)].copy()
        val = df[df["UserId"].isin(val_users)].copy()
    total = len(train) + len(val)
    print(f"     Train users: {len(train_users):,} | rows: {len(train):,} ({len(train) / total * 100:.1f}%)")
    print(f"     Val users:   {len(val_users):,} | rows: {len(val):,} ({len(val) / total * 100:.1f}%)")
    print(f"     Label mean: train={train['Label'].mean():.4f}  val={val['Label'].mean():.4f}")

    # ── 5b. Filter rare items (based on TRAIN frequency only) ──
    if MIN_ITEM_INTERACTIONS > 0:
        with timer("5b. Filter rare items"):
            train_item_counts = train["ItemId"].value_counts()
            valid_items = train_item_counts[train_item_counts >= MIN_ITEM_INTERACTIONS].index
            n_train_before = len(train)
            n_val_before = len(val)
            train = train[train["ItemId"].isin(valid_items)]
            val = val[val["ItemId"].isin(valid_items)]
        n_train_removed = n_train_before - len(train)
        n_val_removed = n_val_before - len(val)
        print(f"     Train items < {MIN_ITEM_INTERACTIONS} (in train): {n_train_removed:,} rows removed")
        print(f"     Val items unseen in train: {n_val_removed:,} rows removed")
        print(f"     Train: {len(train):,}  Val: {len(val):,}")

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
            "n_train_users": int(train["UserId"].nunique()),
            "n_val_users": int(val["UserId"].nunique()),
            "min_item_interactions": MIN_ITEM_INTERACTIONS,
            "hour_bin_size": HOUR_BIN_SIZE,
            "sampling": "stratified_by_hour",
            "split": "user_holdout",
            "label_raw_range": [int(raw_min), int(raw_max)],
            "label_norm": "2*tanh(x/5) -> (0, 2)",
            "label_percentiles": {f"{p}%": round(float(df["Label"].quantile(p / 100)), 4) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
            "timestamp_span_hours": int((df["Timestamp"].max() - df["Timestamp"].min()) / 3600),
            "train_ratio": TRAIN_RATIO,
            "random_seed": RANDOM_SEED,
            # "columns": ["UserId", "ItemId", "Timestamp", "Label"],
            "columns": ["UserId", "ItemId", "Timestamp", "Label", "CategoryId"],
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
