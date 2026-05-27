import warnings
from typing import cast

import pandas as pd


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = df[["user_id", "parent_asin", "rating"]].isnull().sum()
    if missing.any():
        warnings.warn(f"Missing values: {missing[missing > 0].to_dict()}")

    dups = int(df.duplicated(["user_id", "parent_asin"]).sum())
    if dups > 0:
        warnings.warn(f"{dups:,} duplicate (user, item) pairs")

    invalid = int((~df["rating"].between(1, 5)).sum())
    if invalid > 0:
        warnings.warn(f"{invalid:,} ratings outside [1, 5]")

    null_ts = int(df["timestamp"].isnull().sum())
    if null_ts > 0:
        warnings.warn(f"{null_ts:,} null timestamps")

    return df


def clean(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    df = cast(pd.DataFrame, df.dropna(subset=["user_id", "parent_asin", "rating"]))
    df = cast(pd.DataFrame, df[df["rating"].between(1, 5)])
    df = cast(
        pd.DataFrame,
        df.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last"),
    )

    while True:
        prev = len(df)
        u_counts = df.groupby("user_id")["parent_asin"].transform("count")
        df = cast(pd.DataFrame, df[u_counts >= min_interactions])
        i_counts = df.groupby("parent_asin")["user_id"].transform("count")
        df = cast(pd.DataFrame, df[i_counts >= min_interactions])
        if len(df) == prev:
            break

    return df.reset_index(drop=True)


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    df = df.copy()
    user_codes, _ = pd.factorize(df["user_id"])
    item_codes, _ = pd.factorize(df["parent_asin"])
    df["user_idx"] = user_codes
    df["item_idx"] = item_codes
    df["label"] = 1
    num_users = int(df["user_idx"].max()) + 1
    num_items = int(df["item_idx"].max()) + 1
    return df, num_users, num_items


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_idx", "timestamp"]).copy()
    df["_rank"] = df.groupby("user_idx").cumcount(ascending=False)
    test = cast(pd.DataFrame, df[df["_rank"] == 0].drop(columns="_rank").reset_index(drop=True))
    val = cast(pd.DataFrame, df[df["_rank"] == 1].drop(columns="_rank").reset_index(drop=True))
    train = cast(pd.DataFrame, df[df["_rank"] >= 2].drop(columns="_rank").reset_index(drop=True))
    return train, val, test
