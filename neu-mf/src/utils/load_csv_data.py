import numpy as np
import pandas as pd
from pathlib import Path

_CSV_DIR = Path("../data/processs")


def load_csv_data() -> tuple[pd.DataFrame, pd.DataFrame, int, int] | None:
    """Load pre-processed CSVs and encode UserId/ItemId to 0-indexed integers.

    Returns (train_df, val_df, num_users, num_items) or None if files are missing.
    Output DataFrames contain user_idx and item_idx columns (int64, 0-indexed).
    """
    train_path = _CSV_DIR / "train.csv"
    if not train_path.exists():
        return None

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(_CSV_DIR / "val.csv")

    # Build sorted ID arrays from the combined set so val IDs are always covered
    all_users: np.ndarray = np.union1d(train_df["UserId"].to_numpy(), val_df["UserId"].to_numpy())
    all_items: np.ndarray = np.union1d(train_df["ItemId"].to_numpy(), val_df["ItemId"].to_numpy())

    for df in (train_df, val_df):
        df["user_idx"] = np.searchsorted(all_users, df["UserId"].to_numpy()).astype(np.int64)
        df["item_idx"] = np.searchsorted(all_items, df["ItemId"].to_numpy()).astype(np.int64)

    return train_df, val_df, int(len(all_users)), int(len(all_items))
