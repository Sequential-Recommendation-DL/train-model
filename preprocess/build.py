import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import timer, ensure_dir
from config import (
    RAW_DATA,
    PROCESSED_DIR,
    COLUMNS,
    DTYPES,
    TRAIN_PATH,
    VAL_PATH,
    METADATA_PATH,
    MIN_USER_INTERACTIONS,
    MIN_ITEM_INTERACTIONS,
    TIMESTAMP_RANGE,
    TRAIN_RATIO,
    RANDOM_SEED,
)

BEHAVIOR_TYPES = ["pv", "cart", "fav", "buy"]

OUTPUT_COLS = [
    "user_id_enc",
    "item_id_enc",
    "category_id_enc",
    "timestamp",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_pv",
    "is_cart",
    "is_fav",
    "is_buy",
    "user_n_interactions",
    "user_n_buy",
    "user_buy_rate",
    "item_n_interactions",
    "item_n_buy",
    "label",
]


def _add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    df_out = df_out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    dt = pd.to_datetime(df_out["timestamp"], unit="s")
    df_out["hour"] = dt.dt.hour.astype("int8")
    df_out["day_of_week"] = dt.dt.dayofweek.astype("int8")
    df_out["is_weekend"] = (df_out["day_of_week"] >= 5).astype("int8")

    g = df_out.groupby("user_id")
    df_out["user_n_interactions"] = g.cumcount().astype("int32")
    df_out["user_n_buy"] = (g["is_buy"].cumsum() - df_out["is_buy"]).astype("int32")
    df_out["user_buy_rate"] = np.where(
        df_out["user_n_interactions"] > 0,
        df_out["user_n_buy"] / df_out["user_n_interactions"],
        0.0,
    )

    g = df_out.groupby("item_id")
    df_out["item_n_interactions"] = g.cumcount().astype("int32")
    df_out["item_n_buy"] = (g["is_buy"].cumsum() - df_out["is_buy"]).astype("int32")

    return df_out


def run():
    ensure_dir(PROCESSED_DIR)

    with timer("1. Load"):
        df = pd.read_csv(RAW_DATA, names=COLUMNS, dtype=DTYPES)
    n_raw = len(df)
    print(f"  Loaded {n_raw:,} rows")

    with timer("2. Dedup"):
        n_before = len(df)
        df = df.drop_duplicates()
        n_dup = n_before - len(df)
    print(f"  Removed {n_dup:,} dup ({n_dup / n_before * 100:.2f}%)")

    with timer("3. Filter timestamp"):
        t_min, t_max = TIMESTAMP_RANGE
        n_before = len(df)
        df = df[df["timestamp"].between(t_min, t_max)]
        print(f"  Removed {n_before - len(df):,} rows outside range")

    with timer("4. Filter min interactions"):
        n_before = len(df)
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        df = df.reset_index(drop=True)
        print(f"  Removed {n_before - len(df):,} rows (users/items < {MIN_USER_INTERACTIONS})")
    print(f"  Remaining: {len(df):,} rows")

    with timer("5. Label"):
        df["label"] = (df["behavior"] == "buy").astype("int8")
        n_pos = df["label"].sum()
        n_neg = len(df) - n_pos
    print(f"  buy=1: {n_pos:,} ({n_pos / len(df) * 100:.2f}%)")

    with timer("6. One-hot"):
        for beh in BEHAVIOR_TYPES:
            df[f"is_{beh}"] = (df["behavior"] == beh).astype("int8")

    with timer("7. Sort by user + timestamp"):
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    with timer("8. Split"):
        train, val = train_test_split(
            df,
            test_size=1 - TRAIN_RATIO,
            random_state=RANDOM_SEED,
            stratify=df["label"],
        )
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)

    with timer("9. Feature engineering (train)"):
        train = _add_features(train)
    with timer("10. Feature engineering (val)"):
        val = _add_features(val)

    with timer("11. Encode IDs"):
        user_enc = LabelEncoder()
        item_enc = LabelEncoder()
        cat_enc = LabelEncoder()

        train["user_id_enc"] = user_enc.fit_transform(train["user_id"]).astype("int32")
        train["item_id_enc"] = item_enc.fit_transform(train["item_id"]).astype("int32")
        train["category_id_enc"] = cat_enc.fit_transform(train["category_id"]).astype("int16")

        n_val_before = len(val)
        val = val[val["user_id"].isin(user_enc.classes_)].copy()
        val = val[val["item_id"].isin(item_enc.classes_)].copy()
        val = val[val["category_id"].isin(cat_enc.classes_)].copy()
        n_dropped = n_val_before - len(val)
        if n_dropped:
            print(f"  Dropped {n_dropped:,} val rows with unseen IDs")

        val["user_id_enc"] = user_enc.transform(val["user_id"]).astype("int32")
        val["item_id_enc"] = item_enc.transform(val["item_id"]).astype("int32")
        val["category_id_enc"] = cat_enc.transform(val["category_id"]).astype("int16")

    with timer("12. Save"):
        train_out = train[OUTPUT_COLS].copy()
        val_out = val[OUTPUT_COLS].copy()

        train_out.to_csv(TRAIN_PATH, index=False)
        val_out.to_csv(VAL_PATH, index=False)

    total = len(train_out) + len(val_out)
    print(f"  Train: {len(train_out):,} ({len(train_out) / total * 100:.1f}%  pos={train_out['label'].mean():.4f})")
    print(f"  Val:   {len(val_out):,} ({len(val_out) / total * 100:.1f}%  pos={val_out['label'].mean():.4f})")
    print(f"  Train -> {TRAIN_PATH}")
    print(f"  Val   -> {VAL_PATH}")

    metadata = {
        "n_rows_raw": n_raw,
        "n_duplicates": int(n_dup),
        "n_rows_clean": total,
        "n_train": len(train_out),
        "n_val": len(val_out),
        "n_users": len(user_enc.classes_),
        "n_items": len(item_enc.classes_),
        "n_categories": len(cat_enc.classes_),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "positive_ratio": round(n_pos / len(df), 4),
        "min_user_interactions": MIN_USER_INTERACTIONS,
        "min_item_interactions": MIN_ITEM_INTERACTIONS,
        "timestamp_range": list(TIMESTAMP_RANGE),
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "features": [c for c in OUTPUT_COLS if c not in ("label",)],
        "label": "label",
        "note_augmentation": "Apply augmentation to train ONLY. Val must remain raw.",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata -> {METADATA_PATH}")
    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
