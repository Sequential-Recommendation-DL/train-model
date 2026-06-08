from pathlib import Path

RAW_DATA = Path("data/raw/UserBehavior.csv")
PROCESSED_DIR = Path("data/processs")
EDA_DIR = PROCESSED_DIR / "eda"

COLUMNS = ["user_id", "item_id", "category_id", "behavior", "timestamp"]

DTYPES = {
    "user_id": "int32",
    "item_id": "int32",
    "category_id": "int32",
    "behavior": "category",
    "timestamp": "int32",
}

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

TIMESTAMP_RANGE = (1511568000, 1512345600)  # 2017-11-25 00:00:00 UTC → 2017-12-04 00:00:00 UTC

TRAIN_RATIO = 0.9
RANDOM_SEED = 42

SESSION_INACTIVITY_SECONDS = 30 * 60

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
