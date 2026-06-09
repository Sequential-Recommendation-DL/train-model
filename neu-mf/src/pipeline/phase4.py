from src.models.neumf import NeuMF
from src.models.train import train


def train_neumf(device, train_df, val_df, user_pos, num_users, num_items):
    print("\n [4/6] Training NeuMF...")
    model = NeuMF(num_users, num_items)
    print(model.param_summary())

    model, history = train(model, train_df, val_df, user_pos, num_items, device=device)

    best = max(history, key=lambda h: h["HR@10"])
    print(
        f"      Best epoch {best['epoch']:02d}: "
        f"Hit Rate at 10 = {best['HR@10']:.4f}  "
        f"NDCG at 10 = {best['NDCG@10']:.4f}"
    )
    return model, history
