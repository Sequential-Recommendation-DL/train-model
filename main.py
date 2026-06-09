# import os
# import pandas as pd
# import orjson

# from huggingface_hub import hf_hub_download
# from dotenv import load_dotenv
# load_dotenv()

# os.makedirs("data/raw/explicit", exist_ok=True)

# HF_TOKEN = os.getenv("HF_TOKEN")

# REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

# CATEGORIES = {
#     "electronics": "Electronics",
#     "musical_instrument": "Musical_Instruments",
# }

# COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]


# def download_category(name: str, category: str) -> None:
#     path_in_repo = f"raw/review_categories/{category}.jsonl"

#     local_path = z(
#         repo_id=REPO_ID,
#         repo_type="dataset",
#         filename=path_in_repo,
#         token=HF_TOKEN,
#     )

#     print(f"Downloaded {category}.jsonl → {local_path}")

#     out_path = f"data/raw/explicit/{name}.csv"

#     chunk_size = 100_000
#     buffer = []

#     first_write = True
#     total_rows = 0

#     with open(local_path, "rb") as f:
#         for line in f:
#             row = orjson.loads(line)

#             buffer.append({col: row.get(col) for col in COLUMNS})

#             if len(buffer) >= chunk_size:
#                 df = pd.DataFrame(buffer)

#                 df.to_csv(
#                     out_path,
#                     mode="w" if first_write else "a",
#                     header=first_write,
#                     index=False,
#                 )

#                 total_rows += len(df)

#                 print(f"Written {total_rows:,} rows...")

#                 buffer.clear()
#                 first_write = False

#         # write remaining rows
#         if buffer:
#             df = pd.DataFrame(buffer)

#             df.to_csv(
#                 out_path,
#                 mode="w" if first_write else "a",
#                 header=first_write,
#                 index=False,
#             )

#             total_rows += len(df)

#     print(f"Saved {total_rows:,} rows → {out_path}\n")


# for name, category in CATEGORIES.items():
#     print(f"Downloading {name}...")
#     download_category(name, category)

# print("All downloads complete.")

import os
import pandas as pd
import orjson

from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
load_dotenv()

os.makedirs("data/raw/explicit", exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

CATEGORIES = {
    "electronics": "Electronics",
    "musical_instrument": "Musical_Instruments",
}

# [THÊM MỚI] Các cột lấy từ file review (giữ nguyên như cũ)
REVIEW_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]

# [THÊM MỚI] Các cột lấy từ file metadata
META_COLUMNS = ["parent_asin", "price", "brand", "main_category"]


def download_reviews(name: str, category: str) -> None:
    """Giữ nguyên như cũ — download file review"""
    path_in_repo = f"raw/review_categories/{category}.jsonl"

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=path_in_repo,
        token=HF_TOKEN,
    )

    print(f"Đã tải {category}.jsonl → {local_path}")

    out_path = f"data/raw/explicit/{name}_reviews.csv"  # [ĐỔI TÊN] thêm _reviews để phân biệt

    chunk_size = 100_000
    buffer = []
    first_write = True
    total_rows = 0

    with open(local_path, "rb") as f:
        for line in f:
            row = orjson.loads(line)
            buffer.append({col: row.get(col) for col in REVIEW_COLUMNS})

            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)
                df.to_csv(out_path, mode="w" if first_write else "a",
                          header=first_write, index=False)
                total_rows += len(df)
                print(f"  Written {total_rows:,} rows...")
                buffer.clear()
                first_write = False

        if buffer:
            df = pd.DataFrame(buffer)
            df.to_csv(out_path, mode="w" if first_write else "a",
                      header=first_write, index=False)
            total_rows += len(df)

    print(f"Saved {total_rows:,} rows → {out_path}\n")


# [THÊM MỚI] Hàm download metadata
def download_metadata(name: str, category: str) -> None:
    """Download file metadata — có price, brand, main_category"""
    path_in_repo = f"raw/meta_categories/meta_{category}.jsonl"

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=path_in_repo,
        token=HF_TOKEN,
    )

    print(f"Downloaded meta_{category}.jsonl → {local_path}")

    out_path = f"data/raw/explicit/{name}_meta.csv"

    chunk_size = 100_000
    buffer = []
    first_write = True
    total_rows = 0

    with open(local_path, "rb") as f:
        for line in f:
            row = orjson.loads(line)
            buffer.append({col: row.get(col) for col in META_COLUMNS})

            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)
                df.to_csv(out_path, mode="w" if first_write else "a",
                          header=first_write, index=False)
                total_rows += len(df)
                print(f"  Written {total_rows:,} rows...")
                buffer.clear()
                first_write = False

        if buffer:
            df = pd.DataFrame(buffer)
            df.to_csv(out_path, mode="w" if first_write else "a",
                      header=first_write, index=False)
            total_rows += len(df)

    print(f"Saved {total_rows:,} rows → {out_path}\n")


# [THÊM MỚI] Hàm join review + metadata
def merge_and_save(name: str) -> None:
    """Join review với metadata theo parent_asin"""
    review_path = f"data/raw/explicit/{name}_reviews.csv"
    meta_path   = f"data/raw/explicit/{name}_meta.csv"
    out_path    = f"data/raw/explicit/{name}.csv"  # file cuối giữ tên cũ để pipeline không đổi

    print(f"Merging {name}...")
    reviews = pd.read_csv(review_path)
    meta    = pd.read_csv(meta_path).drop_duplicates("parent_asin")

    df = reviews.merge(meta, on="parent_asin", how="left")

    # fill missing price/brand bằng "unknown"
    df["price"]         = df["price"].fillna(-1)       # -1 = không có giá
    df["brand"]         = df["brand"].fillna("unknown")
    df["main_category"] = df["main_category"].fillna("unknown")

    df.to_csv(out_path, index=False)
    print(f"Merged {len(df):,} rows → {out_path}\n")


for name, category in CATEGORIES.items():
    print(f"\n{'='*50}")
    print(f"Downloading reviews: {name}...")
    download_reviews(name, category)

    print(f"Downloading metadata: {name}...")
    download_metadata(name, category)

    merge_and_save(name)

print("All downloads complete.")