
# import warnings
# from typing import cast
# import pandas as pd


# def validate(df: pd.DataFrame) -> pd.DataFrame:
#     missing = df[["user_id", "parent_asin", "rating"]].isnull().sum()
#     if missing.any():
#         warnings.warn(f"Missing values: {missing[missing > 0].to_dict()}")
#     dups = int(df.duplicated(["user_id", "parent_asin"]).sum())
#     if dups > 0:
#         warnings.warn(f"{dups:,} duplicate (user, item) pairs")
#     invalid = int((~df["rating"].between(1, 5)).sum())
#     if invalid > 0:
#         warnings.warn(f"{invalid:,} ratings outside [1, 5]")
#     null_ts = int(df["timestamp"].isnull().sum())
#     if null_ts > 0:
#         warnings.warn(f"{null_ts:,} null timestamps")
#     return df


# def clean(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
#     df = cast(pd.DataFrame, df.dropna(subset=["user_id", "parent_asin", "rating"]))
#     df = cast(pd.DataFrame, df[df["rating"].between(1, 5)])
#     df = cast(
#         pd.DataFrame,
#         df.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last"),
#     )
#     while True:
#         prev = len(df)
#         u_counts = df.groupby("user_id")["parent_asin"].transform("count")
#         df = cast(pd.DataFrame, df[u_counts >= min_interactions])
#         i_counts = df.groupby("parent_asin")["user_id"].transform("count")
#         df = cast(pd.DataFrame, df[i_counts >= min_interactions])
#         if len(df) == prev:
#             break
#     return df.reset_index(drop=True)


# def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
#     df = df.copy()

#     # encode user và item
#     user_codes, _ = pd.factorize(df["user_id"])
#     item_codes, _ = pd.factorize(df["parent_asin"])
#     df["user_idx"] = user_codes
#     df["item_idx"] = item_codes

#     # [THÊM MỚI] encode brand và main_category thành số nguyên
#     if "brand" in df.columns:
#         df["brand"] = df["brand"].fillna("unknown")
#         brand_codes, _ = pd.factorize(df["brand"])
#         df["brand_idx"] = brand_codes
#     else:
#         df["brand_idx"] = 0

#     if "main_category" in df.columns:
#         df["main_category"] = df["main_category"].fillna("unknown")
#         cat_codes, _ = pd.factorize(df["main_category"])
#         df["category_idx"] = cat_codes
#     else:
#         df["category_idx"] = 0

#     # [THÊM MỚI] bucketize price thành 10 khoảng
#     if "price" in df.columns:
#         df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1)
#         df["price_idx"] = pd.qcut(
#             df["price"].clip(lower=0),
#             q=10,
#             labels=False,
#             duplicates="drop"
#         ).fillna(0).astype(int)
#     else:
#         df["price_idx"] = 0

#     # label: rating >= 4 → 1 (thích), < 4 → 0 (không thích)
#     df["label"] = (df["rating"] >= 4).astype(int)

#     num_users = int(df["user_idx"].max()) + 1
#     num_items = int(df["item_idx"].max()) + 1
#     return df, num_users, num_items


# def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     df = df.sort_values(["user_idx", "timestamp"]).copy()
#     df["_rank"] = df.groupby("user_idx").cumcount(ascending=False)
#     test  = cast(pd.DataFrame, df[df["_rank"] == 0].drop(columns="_rank").reset_index(drop=True))
#     val   = cast(pd.DataFrame, df[df["_rank"] == 1].drop(columns="_rank").reset_index(drop=True))
#     train = cast(pd.DataFrame, df[df["_rank"] >= 2].drop(columns="_rank").reset_index(drop=True))
#     return train, val, test
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

    # encode user và item
    user_codes, _ = pd.factorize(df["user_id"])
    item_codes, _ = pd.factorize(df["parent_asin"])
    df["user_idx"] = user_codes
    df["item_idx"] = item_codes

    # encode brand
    # if "brand" in df.columns:
    #     df["brand"] = df["brand"].fillna("unknown")
    #     brand_codes, _ = pd.factorize(df["brand"])
    #     df["brand_idx"] = brand_codes
    # else:
    #     df["brand_idx"] = 0

    # encode main_category
    if "main_category" in df.columns:
        df["main_category"] = df["main_category"].fillna("unknown")
        cat_codes, _ = pd.factorize(df["main_category"])
        df["category_idx"] = cat_codes
    else:
        df["category_idx"] = 0

    # bucketize price thành 10 khoảng
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1)
        df["price_idx"] = pd.qcut(
            df["price"].clip(lower=0),
            q=10,
            labels=False,
            duplicates="drop"
        ).fillna(0).astype(int)
    else:
        df["price_idx"] = 0

    # [THÊM MỚI] tách timestamp thành hour và dayofweek
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        # fallback nếu timestamp tính bằng giây thay vì mili giây
        if dt.isnull().mean() > 0.5:
            dt = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

        df["hour_idx"]      = dt.dt.hour.fillna(0).astype(int)       # 0-23
        df["dayofweek_idx"] = dt.dt.dayofweek.fillna(0).astype(int)  # 0=Mon, 6=Sun
    else:
        df["hour_idx"]      = 0
        df["dayofweek_idx"] = 0

    # label cho implicit feedback (dùng trong ExplicitDataset nếu cần)
    df["label"] = (df["rating"] >= 4).astype(int)
    print("\n===== ENCODE DEBUG =====")
    # print("Brand idx unique:", df["brand_idx"].nunique())
    print("Category idx unique:", df["category_idx"].nunique())
    print("Price idx unique:", df["price_idx"].nunique())
    print("Hour idx unique:", df["hour_idx"].nunique())
    print("Day idx unique:", df["dayofweek_idx"].nunique())
    print("========================\n")

    num_users = int(df["user_idx"].max()) + 1
    num_items = int(df["item_idx"].max()) + 1
    return df, num_users, num_items


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_idx", "timestamp"]).copy()
    df["_rank"] = df.groupby("user_idx").cumcount(ascending=False)
    test  = cast(pd.DataFrame, df[df["_rank"] == 0].drop(columns="_rank").reset_index(drop=True))
    val   = cast(pd.DataFrame, df[df["_rank"] == 1].drop(columns="_rank").reset_index(drop=True))
    train = cast(pd.DataFrame, df[df["_rank"] >= 2].drop(columns="_rank").reset_index(drop=True))
    return train, val, test