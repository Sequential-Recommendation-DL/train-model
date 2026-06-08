from src.data.preprocess import encode, split
from src.utils.save_clean_data import saveCleanData

def encodingAndSplitting(df, cache_key):
    print("\n[2/6] Encoding & splitting...")
    df, num_users, num_items = encode(df)
    print(f" - Users: {num_users:,}  Items: {num_items:,}")
    train_df, val_df, test_df = split(df)
    print(f" - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    saveCleanData(
        cache_key,
        train_df,
        val_df,
        test_df,
        num_users,
        num_items
    );
    return train_df, val_df, test_df, num_users, num_items
