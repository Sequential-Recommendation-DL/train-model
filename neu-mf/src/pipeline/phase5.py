from src.models.predict import evaluate_full

HR_THRESHOLD = 0.60
NDCG_THRESHOLD = 0.35


def evaluate_neumf(device, model, test_df, user_pos, num_items, max_eval_users=2_000):
    print("\n [5/6] Evaluating on test set...")
    metrics = evaluate_full(
        model, test_df, user_pos, num_items,
        device=device, max_users=max_eval_users,
    )

    n_hits = metrics["n_hits_at_k"]
    n_total = metrics["n_eval_users"]
    cm = metrics["confusion_matrix"]
    tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]

    print(
        f"      Hit Rate at 10 = {n_hits}/{n_total} = {metrics['HR@10']:.4f}\n"
        f"      NDCG at 10     = {metrics['NDCG@10']:.4f}\n"
        f"      AUC ROC        = {metrics['AUC_ROC']:.4f}\n"
        f"      Precision      = {metrics['Precision']:.4f}\n"
        f"      Recall         = {metrics['Recall']:.4f}\n"
        f"      F1             = {metrics['F1']:.4f}\n"
        f"      Confusion      : TN={tn}  FP={fp}  FN={fn}  TP={tp}"
    )

    if metrics["HR@10"] < HR_THRESHOLD or metrics["NDCG@10"] < NDCG_THRESHOLD:
        print(
            f"\n      WARNING: below acceptance thresholds "
            f"(Hit Rate at 10 >= {HR_THRESHOLD}, NDCG at 10 >= {NDCG_THRESHOLD}). "
            "Check data quality or tune hyperparameters."
        )

    return metrics
