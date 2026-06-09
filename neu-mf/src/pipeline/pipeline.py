import numpy as np

from src.pipeline.phase1 import loading_and_clean_data
from src.pipeline.phase2 import encodingAndSplitting
from src.pipeline.phase3 import build_training_features
from src.pipeline.phase4 import train_neumf
from src.pipeline.phase5 import evaluate_neumf
from src.pipeline.phase6 import save_results
from src.utils.check_gpu import check_gpu
from src.utils.min_interactions import MinInteractions
from src.utils.load_cache_data import loadCacheData


def run_pipeline(
    max_users: int | None = 250_000,
    model_path: str = "../models/neumf_best.pt",
):
    device = check_gpu()
    min_interactions = MinInteractions
    cache_key = f"mi{min_interactions.value}" + (f"_u{max_users}" if max_users else "")

    cached = loadCacheData(cache_key)
    if cached:
        train_df, val_df, test_df, num_users, num_items = cached
        print(f"\n [2/6] Loaded processed data from cache (key={cache_key})")
        print(f"       Users: {num_users:,}  Items: {num_items:,}")
        print(f"       Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    else:
        df = loading_and_clean_data(min_interactions)
        if max_users is not None:
            unique_users = np.asarray(df["user_id"].unique())
            if len(unique_users) > max_users:
                rng = np.random.default_rng(42)
                sampled = rng.choice(unique_users, size=max_users, replace=False)
                df = df[df["user_id"].isin(sampled)].reset_index(drop=True)
                print(f"\n       Pre-split subset: {max_users:,} users — {len(df):,} interactions")
        train_df, val_df, test_df, num_users, num_items = encodingAndSplitting(df, cache_key)

    user_pos = build_training_features(train_df)
    model, history = train_neumf(device, train_df, val_df, user_pos, num_users, num_items)
    metrics = evaluate_neumf(device, model, test_df, user_pos, num_items)
    save_results(model, metrics, history, num_users, num_items, model_path)

    print("\nDone.")


if __name__ == "__main__":
    run_pipeline()
