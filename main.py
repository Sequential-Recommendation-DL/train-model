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

COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]


def download_category(name: str, category: str) -> None:
    path_in_repo = f"raw/review_categories/{category}.jsonl"

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=path_in_repo,
        token=HF_TOKEN,
    )

    print(f"Downloaded {category}.jsonl → {local_path}")

    out_path = f"data/raw/explicit/{name}.csv"

    chunk_size = 100_000
    buffer = []

    first_write = True
    total_rows = 0

    with open(local_path, "rb") as f:
        for line in f:
            row = orjson.loads(line)

            buffer.append({col: row.get(col) for col in COLUMNS})

            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)

                df.to_csv(
                    out_path,
                    mode="w" if first_write else "a",
                    header=first_write,
                    index=False,
                )

                total_rows += len(df)

                print(f"Written {total_rows:,} rows...")

                buffer.clear()
                first_write = False

        # write remaining rows
        if buffer:
            df = pd.DataFrame(buffer)

            df.to_csv(
                out_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            total_rows += len(df)

    print(f"Saved {total_rows:,} rows → {out_path}\n")


for name, category in CATEGORIES.items():
    print(f"Downloading {name}...")
    download_category(name, category)

print("All downloads complete.")