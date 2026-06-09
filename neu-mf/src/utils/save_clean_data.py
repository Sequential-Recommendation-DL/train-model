import json
from pathlib import Path

def saveCleanData(
    key,
    train_df,
    val_df,
    test_df,
    num_users,
    num_items,
    PROCESSED_DIR = Path("../data/processed")
):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_pickle(PROCESSED_DIR / f"{key}_train.pkl")
    val_df.to_pickle(PROCESSED_DIR / f"{key}_val.pkl")
    test_df.to_pickle(PROCESSED_DIR / f"{key}_test.pkl")
    (PROCESSED_DIR / f"{key}_meta.json").write_text(
        json.dumps({"num_users": num_users, "num_items": num_items})
    )
    print(f" - Saved processed data → {PROCESSED_DIR}")
