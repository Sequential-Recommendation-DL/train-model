# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
torch.set_float32_matmul_precision('medium')
import pandas as pd
from torch.utils.data import DataLoader

from config import config
from data.preprocessing import DataPreprocessor
from data.dataset import NeuMFTrainDataset, NeuMFDataset
from models.neumf import NeuMF
from trainers.trainer import ModelTrainer
from utils.callbacks import MetricsHistory
from utils.metrics import evaluate_ranking
from evaluation.evaluator import Evaluator

# Set random seeds
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(config.GLOBAL_SEED)

def worker_init_fn(worker_id):
    np.random.seed((config.GLOBAL_SEED + worker_id + torch.initial_seed()) % 2**31)

def build_ranking_fixture(df, interacted_dict, num_items, n_neg, n_samples, seed):
    """Build ranking evaluation fixture"""
    rng = np.random.default_rng(seed)
    pairs = df[['user_idx', 'item_idx']].values
    if n_samples is not None and n_samples < len(pairs):
        idx = rng.choice(len(pairs), n_samples, replace=False)
        pairs = pairs[idx]
    
    users_mat = np.zeros((len(pairs), n_neg + 1), dtype=np.int64)
    items_mat = np.zeros((len(pairs), n_neg + 1), dtype=np.int64)
    
    for i, (u_idx, pos_item) in enumerate(pairs):
        interacted = interacted_dict.get(u_idx, set())
        negs = []
        while len(negs) < n_neg:
            cands = rng.integers(0, num_items, n_neg * 2)
            negs.extend(c for c in cands if c not in interacted)
        users_mat[i, :] = u_idx
        items_mat[i, 0] = pos_item
        items_mat[i, 1:] = negs[:n_neg]
    
    return torch.from_numpy(users_mat), torch.from_numpy(items_mat)

def fast_negative_sampling(df, interacted_dict, num_items, num_neg, seed, desc):
    """Fast negative sampling for validation"""
    rng = np.random.default_rng(seed)
    users_out, items_out = [], []
    
    for u_idx, group in df.groupby('user_idx'):
        interacted = interacted_dict.get(u_idx, set())
        n_needed = len(group) * num_neg
        negs = []
        while len(negs) < n_needed:
            candidates = rng.integers(0, num_items, n_needed * 2)
            negs.extend(c for c in candidates if c not in interacted)
        users_out.extend([u_idx] * n_needed)
        items_out.extend(negs[:n_needed])
    
    return pd.DataFrame({'user_idx': users_out, 'item_idx': items_out, 'label': 0})

def main():
    print("="*60)
    print("NEUMF RECOMMENDATION SYSTEM (MSE Loss + Adam)")
    print(f"Run folder: {config.RESULT_PATH}")
    print("="*60)
    
    # Step 1: Data preprocessing
    print("\n[1/6] Loading and preprocessing data...")
    preprocessor = DataPreprocessor(config)
    train_df, val_df = preprocessor.load_data()
    preprocessor.build_mappings(train_df)
    train_df, val_df = preprocessor.transform_data(train_df, val_df)
    
    num_users = len(preprocessor.user2idx)
    num_items = len(preprocessor.item2idx)
    print(f"✓ Users: {num_users:,} | Items: {num_items:,}")

    # Step 2: Build popularity distribution
    print("\n[3/6] Building popularity distribution...")
    pop_cum = preprocessor.build_popularity_distribution(train_df, num_items)
    
    # Step 4: Negative sampling for validation
    print("\n[4/6] Negative sampling for validation...")
    user_interacted_train = preprocessor.get_interacted_items(train_df)
    all_user_interacted = preprocessor.get_interacted_items(train_df, val_df)
    
    val_neg_df = fast_negative_sampling(val_df, all_user_interacted, num_items,
                                        config.NUM_NEGATIVES, config.GLOBAL_SEED + 1, "")
    val_pos_df = val_df[['user_idx', 'item_idx', 'label']].copy()
    val_final_df = pd.concat([val_pos_df, val_neg_df], ignore_index=True)
    val_final_df = val_final_df.sample(frac=1, random_state=config.GLOBAL_SEED).reset_index(drop=True)
    
    # Step 5: Create datasets and dataloaders
    print("\n[5/6] Creating datasets and dataloaders...")
    train_dataset = NeuMFTrainDataset(train_df, user_interacted_train, num_items,
                                      config.NUM_NEGATIVES, pop_cum)
    val_dataset = NeuMFDataset(val_final_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=4, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=4, persistent_workers=True)
    
    # Step 6: Build ranking fixtures
    print("\n[6/6] Building ranking evaluation fixtures...")
    eval_users_epoch, eval_items_epoch = build_ranking_fixture(
        val_df, all_user_interacted, num_items, config.N_EVAL_NEG,
        config.N_EVAL_SAMPLES, config.GLOBAL_SEED + 2)
    eval_users_full, eval_items_full = build_ranking_fixture(
        val_df, all_user_interacted, num_items, config.N_EVAL_NEG,
        None, config.GLOBAL_SEED + 3)
    
    # Initialize model
    model = NeuMF(config, num_users, num_items)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Metrics callback
    metrics_history = MetricsHistory(eval_users_epoch, eval_items_epoch, config.TOP_K, evaluate_ranking)
    
    # Train model
    trainer_obj = ModelTrainer(config, model, train_dataloader, val_dataloader,
                               metrics_history, config.RESULT_PATH)
    trainer, training_time = trainer_obj.train()
    
    # Save results
    print("\nSaving results...")
    csv_path, curves_path = metrics_history.save(config.RESULT_PATH)
    print(f"✓ Training history: {csv_path}")
    print(f"✓ Learning curves: {curves_path}")
    
    # Load best checkpoint
    best_ckpt = trainer_obj.checkpoint_callback.best_model_path
    if best_ckpt:
        print(f"Loading best checkpoint: {best_ckpt}")
        model = NeuMF.load_from_checkpoint(best_ckpt, config=config,
                                           num_users=num_users, num_items=num_items)
    
    # Final evaluation
    print("\nFull ranking evaluation on validation set...")
    hr_full, ndcg_full = evaluate_ranking(model, eval_users_full, eval_items_full, config.TOP_K)
    print(f"✓ NeuMF HR@{config.TOP_K}: {hr_full:.4f} | NDCG@{config.TOP_K}: {ndcg_full:.4f}")
    
    # Baseline ItemPop
    item_pop = np.zeros(num_items)
    pop_counts = train_df.groupby('item_idx').size()
    item_pop[pop_counts.index] = pop_counts.values
    
    items_np = eval_items_full.numpy()
    pop_scores = item_pop[items_np]
    pop_rank = (pop_scores > pop_scores[:, :1]).sum(axis=1) + 1
    pop_hr = float((pop_rank <= config.TOP_K).mean())
    pop_ndcg = float(np.where(pop_rank <= config.TOP_K, 1.0 / np.log2(pop_rank + 1), 0.0).mean())
    print(f"✓ ItemPop HR@{config.TOP_K}: {pop_hr:.4f} | NDCG@{config.TOP_K}: {pop_ndcg:.4f}")
    
    # Detailed evaluation
    evaluator = Evaluator(model, val_dataloader, config.RESULT_PATH)
    eval_metrics = evaluator.evaluate()
    
    # Save summary
    summary_path = os.path.join(config.RESULT_PATH, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NEUMF TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Run: {config.RUN_NAME}\n\n")
        f.write("HYPERPARAMETERS:\n")
        f.write(f"  - Optimizer: AdamW with decoupled weight decay\n")
        f.write(f"  - LR: {config.LEARNING_RATE} (warmup {config.WARMUP_EPOCHS} epochs + cosine decay)\n")
        f.write(f"  - Loss: MSE (Mean Squared Error)\n")
        f.write(f"  - Embedding dim: {config.EMBEDDING_DIM}\n")
        f.write(f"  - Dropout: {config.DROPOUT_RATE}\n")
        f.write(f"  - Weight decay: linear={config.WEIGHT_DECAY}, emb={config.EMB_WEIGHT_DECAY}\n\n")
        f.write(f"TRAINING TIME: {training_time/60:.2f} minutes\n\n")
        f.write("FINAL METRICS:\n")
        f.write(f"  - HR@{config.TOP_K}: {hr_full:.4f}\n")
        f.write(f"  - NDCG@{config.TOP_K}: {ndcg_full:.4f}\n")
        f.write(f"  - ROC AUC: {eval_metrics['roc_auc']:.4f}\n")
        f.write(f"  - PR AUC (AP): {eval_metrics['pr_auc']:.4f}\n")
        f.write(f"  - F1 Score (optimal): {eval_metrics['f1_best']:.4f}\n")
        f.write(f"  - Optimal threshold: {eval_metrics['best_threshold']:.4f}\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print("\n" + "="*60)
    print(f"ALL DONE! Results in: {config.RESULT_PATH}")
    print("="*60)
    print("\nTo view training curves, run:")
    print(f"tensorboard --logdir={config.RESULT_PATH}/tensorboard_logs")

if __name__ == "__main__":
    main()
