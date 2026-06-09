from src.features.build_features import NCFDataset, build_user_pos, negative_sample

def build_training_features(
    device,
    train_df,
    num_items,
    batch_size,
    num_neg_train,
):
    print("\n[3/6] Building training features...")
    user_pos = build_user_pos(train_df)
    '''
    train_sampled = negative_sample(train_df, num_items, user_pos, num_neg=num_neg_train)
    pin = device == "cuda"
    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        NCFDataset(train_sampled), batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=pin, persistent_workers=True,
    )
    print(f"      Training samples: {len(train_sampled):,}  Batches/epoch: {len(train_loader):,}")
'''

