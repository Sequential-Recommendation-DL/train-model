import shutil
from pathlib import Path

import kagglehub

RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "UserBehavior.csv"


def run():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists():
        resp = input(f"{RAW_FILE} already exists. Download again? (y/N): ")
        if resp.lower() != "y":
            print("Skipped.")
            return

    print("Downloading UserBehavior dataset from Kaggle...")
    path = kagglehub.dataset_download("marwa80/userbehavior")
    src = Path(path) / "UserBehavior.csv"

    print(f"Copying {src} -> {RAW_FILE} ...")
    shutil.copy2(src, RAW_FILE)
    print(f"Done. ({RAW_FILE})")


if __name__ == "__main__":
    run()
