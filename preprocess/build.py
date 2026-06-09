import json
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


def run():
    ensure_dir(PROCESSED_DIR)

    with timer("1. Load"):
        df = pd.read_csv(RAW_DATA, names=COLUMNS, dtype=DTYPES)
    print(f"  Loaded {len(df):,} rows\n")

    with timer("2. Dedup + filter timestamp"):
        n_before = len(df)
        df = df.drop_duplicates()
        t_min, t_max = TIMESTAMP_RANGE
        df = df[df["timestamp"].between(t_min, t_max)]
        print(f"  Kept {len(df):,} / {n_before:,} rows (dups + timestamp)\n")

    with timer("3. Score behavior"):
        score_map = {"pv": 1, "fav": 2, "cart": 3, "buy": 4}
        df["label"] = df["behavior"].map(score_map).astype("int8")
    print(f"  Score distribution:\n{df['label'].value_counts().sort_index().to_string()}\n")

    with timer("4. Group by (user, item)"):
        g = df.groupby(["user_id", "item_id"], as_index=False)
        df = g.agg(Timestamp=("timestamp", "max"), Label=("label", "sum"))
        df.columns = ["UserId", "ItemId", "Timestamp", "Label"]
    print(f"  After groupby: {len(df):,} rows\n")
    print(f"  Label distribution:\n{df['Label'].value_counts().sort_index().head(20).to_string()}\n")

    with timer("5. Split + save"):
        train, val = train_test_split(
            df,
            test_size=1 - TRAIN_RATIO,
            random_state=RANDOM_SEED,
        )

        train.to_csv(TRAIN_PATH, index=False)
        val.to_csv(VAL_PATH, index=False)

    total = len(train) + len(val)
    print(f"  Train: {len(train):,} ({len(train) / total * 100:.1f}%) \u2192 {TRAIN_PATH}")
    print(f"  Val:   {len(val):,} ({len(val) / total * 100:.1f}%) \u2192 {VAL_PATH}\n")

    with timer("6. Save metadata"):
        score_dist = df["Label"].value_counts().sort_index()
        metadata = {
            "n_rows_raw": n_before,
            "n_rows_clean": total,
            "n_train": len(train),
            "n_val": len(val),
            "score_distribution": {
                str(k): int(v) for k, v in score_dist.items()
            },
            "timestamp_range": list(TIMESTAMP_RANGE),
            "train_ratio": TRAIN_RATIO,
            "random_seed": RANDOM_SEED,
            "columns": ["UserId", "ItemId", "Timestamp", "Label"],
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
    print(f"  Metadata \u2192 {METADATA_PATH}")

    print(f"\n{'=' * 40}")
    print("  Done!")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    run()
