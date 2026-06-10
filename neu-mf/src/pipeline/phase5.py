from src.models.predict import evaluate


def evaluate_neumf(device, model, val_df):
    print("\n [3/4] Evaluating on validation set...")
    metrics = evaluate(model, val_df, device=device)
    print(f"      AUC-ROC = {metrics['AUC_ROC']:.4f}")
    return metrics
