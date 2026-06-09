import os
import pandas as pd

CATEGORIES = ("electronics.csv", "musical_instrument.csv")


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"user_id": str, "parent_asin": str})


def load_all(data_dir: str = "data/raw/explicit") -> pd.DataFrame:
    frames = []
    for fname in CATEGORIES:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            frames.append(load_csv(path))
        else:
            print(f"Warning: {path} not found, skipping.")
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)