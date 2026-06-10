from src.pipeline.phase1 import load_data
from src.pipeline.phase4 import train_neumf
from src.pipeline.phase5 import evaluate_neumf
from src.pipeline.phase6 import save_results
from src.utils.check_gpu import check_gpu
from src.utils.config import load_config


def run_pipeline(model_path: str | None = None):
    cfg = load_config()
    device = check_gpu()

    train_df, val_df, num_users, num_items = load_data()
    model, history = train_neumf(device, train_df, val_df, num_users, num_items, cfg)
    metrics = evaluate_neumf(device, model, val_df)
    save_results(model, metrics, history, num_users, num_items, model_path or cfg["pipeline"]["model_path"])

    print("\nDone.")


if __name__ == "__main__":
    run_pipeline()
