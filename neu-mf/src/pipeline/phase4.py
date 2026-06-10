from src.models.neumf import NeuMF
from src.models.train import train


def train_neumf(device, train_df, val_df, num_users, num_items, cfg):
    print("\n [2/4] Training NeuMF...")
    m = cfg["model"]
    t = cfg["training"]

    model = NeuMF(num_users, num_items, gmf_dim=m["gmf_dim"], mlp_dim=m["mlp_dim"], dropout=m["dropout"])
    print(model.param_summary())

    model, history = train(
        model, train_df, val_df, device=device,
        epochs=t["epochs"], lr=t["lr"],
        weight_decay=t["weight_decay"], patience=t["patience"], batch_size=t["batch_size"],
    )

    best = min(history, key=lambda h: h["val_loss"])
    print(f"      Best epoch {best['epoch']:02d}: val_loss = {best['val_loss']:.4f}")
    return model, history
