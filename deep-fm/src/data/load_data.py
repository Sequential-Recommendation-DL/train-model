# import os
# import pandas as pd

# CATEGORIES = ("electronics.csv", "musical_instrument.csv")


# def load_csv(path: str) -> pd.DataFrame:
#     return pd.read_csv(path, dtype={"user_id": str, "parent_asin": str})


# def load_all(data_dir: str = "data/raw/explicit") -> pd.DataFrame:
#     frames = []
#     for fname in CATEGORIES:
#         path = os.path.join(data_dir, fname)
#         if os.path.exists(path):
#             frames.append(load_csv(path))
#         else:
#             print(f"Warning: {path} not found, skipping.")
#     if not frames:
#         raise FileNotFoundError(f"No CSV files found in {data_dir}")
#     return pd.concat(frames, ignore_index=True)
import os
import pandas as pd

# Taobao UserBehavior — 1 file duy nhất, không có header
TAOBAO_FILE = "UserBehavior.csv"

COLUMNS = ["user_id", "item_id", "category_id", "behavior", "timestamp"]
DTYPES  = {
    "user_id":     "int32",
    "item_id":     "int32",
    "category_id": "int32",
    "behavior":    "category",
    "timestamp":   "int32",
}


def load_all(data_dir: str = "../data/raw") -> pd.DataFrame:
    path = os.path.join(data_dir, TAOBAO_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file Taobao tại: {path}\n"
            f"Tải về tại: https://tianchi.aliyun.com/dataset/649"
        )

    print(f"      Đọc file: {path}")
    df = pd.read_csv(path, names=COLUMNS, dtype=DTYPES, header=None)
    print(f"      Đã đọc: {len(df):,} rows")
    return df