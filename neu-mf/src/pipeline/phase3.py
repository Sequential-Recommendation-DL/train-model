from src.features.build_features import build_user_pos

def build_training_features(train_df):
    print("\n [3/6] Building training features...")
    user_pos = build_user_pos(train_df)
    print(f"      Unique users with interactions: {len(user_pos):,}")
    return user_pos

