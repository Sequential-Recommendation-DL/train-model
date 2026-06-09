import os
import pandas as pd

categories = ("electronics.csv", "musical_instrument.csv")


def load_csv(path):
    return pd.read_csv(path, dtype={"user_id": str, "parent_asin": str})


def load_all(data_dir= "../data/raw/explicit"):
    frames = []
    for category in categories:
        path = os.path.join(data_dir, category)
        if os.path.exists(path):
            df = load_csv(path)
            df["_category"] = category 
            frames.append(df)
        else:
            print(f"Warning: {path} not found, skipping.")
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)
