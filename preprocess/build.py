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
    SESSION_INACTIVITY_SECONDS,
)

BEHAVIOR_TYPES = ["pv", "cart", "fav", "buy"]

OUTPUT_COLS = [
    "user_id_enc",
    "item_id_enc",
    "category_id_enc",
    "timestamp",
    "is_pv",
    "is_cart",
    "is_fav",
    "is_buy",
    "hour",
    "day_of_week",
    "is_weekend",
    "position_in_session",
    "user_n_pv",
    "user_n_cart",
    "user_n_fav",
    "user_n_buy",
    "user_n_interactions",
    "user_buy_rate",
    "item_n_pv",
    "item_n_cart",
    "item_n_fav",
    "item_n_buy",
    "cat_n_interactions",
    "cat_n_buy",
    "cat_buy_rate",
    "ui_n_pv",
    "ui_n_cart",
    "ui_n_fav",
    "ui_n_buy",
    "ui_has_bought",
    "label",
]


def run():
    ensure_dir(PROCESSED_DIR)

    # ── Step 1: Load ──
    with timer("1. Load data"):
        df = pd.read_csv(RAW_DATA, names=COLUMNS, dtype=DTYPES)
    n_raw = len(df)
    print(f"  Loaded {n_raw:,} rows")

    # ── Step 2: Deduplicate ──
    with timer("2. Deduplicate"):
        before = len(df)
        df = df.drop_duplicates()
        n_dup = before - len(df)
    print(f"  Removed {n_dup:,} duplicates ({n_dup / before * 100:.2f}%)")

    # ── Step 2b: Filter timestamps ──
    with timer("2b. Filter timestamp range"):
        t_min, t_max = TIMESTAMP_RANGE
        before_ts = len(df)
        df = df[df["timestamp"].between(t_min, t_max)]
        n_ts_out = before_ts - len(df)
    print(f"  Removed {n_ts_out:,} rows outside timestamp range ({n_ts_out / before_ts * 100:.2f}%)")

    # ── Step 3: Filter ──
    with timer("3. Filter by min interactions"):
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        before_filter = len(df)
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        df = df.reset_index(drop=True)
        n_filtered = before_filter - len(df)
    print(f"  Removed {n_filtered:,} rows from filtering ({n_filtered / before_filter * 100:.2f}%)")
    print(f"  Remaining: {len(df):,} rows")

    # ── Step 4: Label ──
    with timer("4. Create label"):
        df["label"] = (df["behavior"] == "buy").astype("int8")
        n_pos = df["label"].sum()
        n_neg = len(df) - n_pos
    print(f"  Positive (buy): {n_pos:,} ({n_pos / len(df) * 100:.2f}%)")
    print(f"  Negative:       {n_neg:,} ({n_neg / len(df) * 100:.2f}%)")

    # ── Step 5: Behavior dummies ──
    with timer("5. Behavior one-hot"):
        for beh in BEHAVIOR_TYPES:
            df[f"is_{beh}"] = (df["behavior"] == beh).astype("int8")

    # ── Step 6: Sort for cumulative features ──
    with timer("6. Sort by user + timestamp"):
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # ── Step 7: Time features ──
    with timer("7. Time features"):
        dt = pd.to_datetime(df["timestamp"], unit="s")
        df["hour"] = dt.dt.hour.astype("int8")
        df["day_of_week"] = dt.dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    # ── Step 8: Session features ──
    with timer("8. Session features"):
        time_since_last = df.groupby("user_id")["timestamp"].diff()
        new_session = (time_since_last > SESSION_INACTIVITY_SECONDS) | time_since_last.isna()
        df["session_id"] = new_session.groupby(df["user_id"]).cumsum()
        df["position_in_session"] = (
            df.groupby(["user_id", "session_id"]).cumcount().astype("int16")
        )

    # ── Step 9: User cumulative features ──
    with timer("9. User cumulative features"):
        group_user = df.groupby("user_id")
        for beh in BEHAVIOR_TYPES:
            df[f"user_n_{beh}"] = group_user[f"is_{beh}"].cumsum() - df[f"is_{beh}"]
        df["user_n_interactions"] = group_user.cumcount()
        df["user_buy_rate"] = np.where(
            df["user_n_interactions"] > 0,
            df["user_n_buy"] / df["user_n_interactions"],
            0.0,
        )

    # ── Step 10: Item cumulative features ──
    with timer("10. Item cumulative features"):
        group_item = df.groupby("item_id")
        for beh in BEHAVIOR_TYPES:
            df[f"item_n_{beh}"] = group_item[f"is_{beh}"].cumsum() - df[f"is_{beh}"]

    # ── Step 11: Category cumulative features ──
    with timer("11. Category cumulative features"):
        group_cat = df.groupby("category_id")
        df["cat_n_interactions"] = group_cat.cumcount()
        df["cat_n_buy"] = group_cat["is_buy"].cumsum() - df["is_buy"]
        df["cat_buy_rate"] = np.where(
            df["cat_n_interactions"] > 0,
            df["cat_n_buy"] / df["cat_n_interactions"],
            0.0,
        )

    # ── Step 12: User-Item cumulative features ──
    with timer("12. User-Item cumulative features"):
        group_ui = df.groupby(["user_id", "item_id"])
        for beh in BEHAVIOR_TYPES:
            df[f"ui_n_{beh}"] = group_ui[f"is_{beh}"].cumsum() - df[f"is_{beh}"]
        df["ui_has_bought"] = (df["ui_n_buy"] > 0).astype("int8")

    # ── Step 13: Encode categorical IDs ──
    with timer("13. Encode IDs"):
        user_enc = LabelEncoder()
        item_enc = LabelEncoder()
        cat_enc = LabelEncoder()
        df["user_id_enc"] = user_enc.fit_transform(df["user_id"]).astype("int32")
        df["item_id_enc"] = item_enc.fit_transform(df["item_id"]).astype("int32")
        df["category_id_enc"] = cat_enc.fit_transform(df["category_id"]).astype("int16")

    # ── Step 14: Select output columns ──
    with timer("14. Select columns"):
        out = df[OUTPUT_COLS].copy()

    # ── Step 15: Train/Validation split ──
    with timer("15. Train/validation split"):
        train, val = train_test_split(
            out,
            test_size=1 - TRAIN_RATIO,
            random_state=RANDOM_SEED,
            stratify=out["label"],
        )
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
    print(f"  Train: {len(train):,} ({len(train) / len(out) * 100:.1f}%)")
    print(f"  Val:   {len(val):,} ({len(val) / len(out) * 100:.1f}%)")
    n_pos_train = train["label"].sum()
    n_pos_val = val["label"].sum()
    print(f"  Train label=1: {n_pos_train:,} ({n_pos_train / len(train) * 100:.2f}%)")
    print(f"  Val   label=1: {n_pos_val:,} ({n_pos_val / len(val) * 100:.2f}%)")

    # ── Step 16: Save ──
    with timer("16. Save"):
        train.to_csv(TRAIN_PATH, index=False)
        val.to_csv(VAL_PATH, index=False)
    print(f"  Train → {TRAIN_PATH}")
    print(f"  Val   → {VAL_PATH}")

    # ── Step 17: Metadata ──
    metadata = {
        "n_rows_raw": n_raw,
        "n_duplicates": int(n_dup),
        "n_rows_after_filter": len(out),
        "n_train": len(train),
        "n_val": len(val),
        "n_users": int(df["user_id"].nunique()),
        "n_items": int(df["item_id"].nunique()),
        "n_categories": int(df["category_id"].nunique()),
        "n_users_encoded": len(user_enc.classes_),
        "n_items_encoded": len(item_enc.classes_),
        "n_categories_encoded": len(cat_enc.classes_),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "positive_ratio": round(float(n_pos / len(df)), 4),
        "min_user_interactions": MIN_USER_INTERACTIONS,
        "min_item_interactions": MIN_ITEM_INTERACTIONS,
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "session_inactivity_seconds": SESSION_INACTIVITY_SECONDS,
        "output_columns": OUTPUT_COLS,
        "feature_columns": [c for c in OUTPUT_COLS if c not in ("label",)],
        "label_column": "label",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata → {METADATA_PATH}")

    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
