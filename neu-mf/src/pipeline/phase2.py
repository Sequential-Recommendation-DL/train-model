import pandas as pd

from src.features.build_features import build_user_pos


def prepare_for_training(train_df, val_df=None):
    print("\n [2/6] Building user interaction history...")
    combined = pd.concat([train_df, val_df], ignore_index=True) if val_df is not None else train_df
    user_pos = build_user_pos(combined)
    print(f"       Unique users with interactions: {len(user_pos):,}")
    return user_pos
