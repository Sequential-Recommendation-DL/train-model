import json
import pandas as pd
from pathlib import Path

def loadCacheData(
    key,
    PROCESSED_DIR = Path("data/processed")
):
    meta_file = PROCESSED_DIR / f"{key}_meta.json"
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text())
    try:
        return (
            pd.read_pickle(PROCESSED_DIR / f"{key}_train.pkl"),
            pd.read_pickle(PROCESSED_DIR / f"{key}_val.pkl"),
            pd.read_pickle(PROCESSED_DIR / f"{key}_test.pkl"),
            meta["num_users"],
            meta["num_items"],
        )
    except Exception:
        return None


