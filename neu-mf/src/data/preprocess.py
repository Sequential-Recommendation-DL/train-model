import warnings
import pandas as pd
from typing import cast

def checkMissValues(df):
    print("\n - Check missing values: ")
    missing = df[["user_id", "parent_asin", "rating", "timestamp"]].isnull().sum()
    print(missing)
    if missing.any():
        warnings.warn(f"Missing values: {missing[missing > 0].to_dict()}")

def checkDuplicateUserProductRatings(df):
    print("\n - Check duplicate pairs: ")
    dups = int(df.duplicated(["user_id", "parent_asin"]).sum())
    if dups > 0:
        print(f"{dups:,} duplicate (user, item) pairs")

def checkInvalidRating(df):
    print("\n - Check invalid rating: ")
    invalid = int((~df["rating"].between(1, 5)).sum())
    if invalid > 0:
        print(f"{invalid:,} ratings outside [1, 5]")

def cleanNullRows(df):
    print("\n - Clean Null Values...")
    df = cast(
       pd.DataFrame, 
       df.dropna(
         subset=[
            "user_id",
            "parent_asin",
            "rating",
            "timestamp",
         ]
       )
    )
    return df

def cleanDuplicateRows(df):
    print("\n - Clean Duplicate...")
    df = df.drop_duplicates(
        subset=["user_id", "parent_asin"],
        keep="last"
    )
    return df
   
def clipRating(df):
    print("\n - Clip Rating...")
    df["rating"] = df["rating"].clip(1, 5)
    return df

def filterActiveUsers(df, min_interactions = 1):
    print("\n - Filter Active User...")
    while True:
        prev = len(df)
        u_counts = df.groupby("user_id")["parent_asin"].transform("count")
        df = cast(pd.DataFrame, df[u_counts >= min_interactions])

        i_counts = df.groupby("parent_asin")["user_id"].transform("count")
        df = cast(pd.DataFrame, df[i_counts >= min_interactions])

        if len(df) == prev:
            break

    return df

def validate(df):
    checkMissValues(df)
    checkDuplicateUserProductRatings(df)
    checkInvalidRating(df)
    return df

def clean(df, mi):
    df = cleanNullRows(df)
    df = cleanDuplicateRows(df)
    df = clipRating(df)
    df = filterActiveUsers(df, mi.value)
    return df.reset_index(drop=True)

def encode(df):
    df = df.copy()
    user_codes, _ = pd.factorize(df["user_id"])
    item_codes, _ = pd.factorize(df["parent_asin"])
    df["user_idx"] = user_codes
    df["item_idx"] = item_codes
    df["label"] = 1
    num_users = int(df["user_idx"].max()) + 1
    num_items = int(df["item_idx"].max()) + 1
    return df, num_users, num_items

def split(df):
    df = df.sort_values(["user_idx", "timestamp"]).copy()
    df["_rank"] = df.groupby("user_idx").cumcount(ascending=False)
    test = cast(
        pd.DataFrame, df[df["_rank"] == 0]
          .drop(columns="_rank").reset_index(drop=True)
    )
    val = cast(
        pd.DataFrame, df[df["_rank"] == 1]
          .drop(columns="_rank").reset_index(drop=True)
    )
    train = cast(
        pd.DataFrame, df[df["_rank"] >= 2]
          .drop(columns="_rank").reset_index(drop=True)
    )
    return train, val, test
