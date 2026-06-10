from src.utils.load_csv_data import load_csv_data


def load_data():
    print("\n [1/4] Loading pre-processed data...")
    result = load_csv_data()
    if result is None:
        raise FileNotFoundError(
            "No processed data found in data/processs/. "
            "Run the preprocessing script first."
        )
    train_df, val_df, num_users, num_items = result
    print(f"       Users: {num_users:,}  Items: {num_items:,}")
    print(f"       Train: {len(train_df):,}  Val: {len(val_df):,}")
    return train_df, val_df, num_users, num_items
