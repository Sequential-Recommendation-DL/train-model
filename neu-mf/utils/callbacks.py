# utils/callbacks.py
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import os

class MetricsHistory(pl.Callback):
    """Single Responsibility: Track and save training metrics"""
    
    def __init__(self, eval_users, eval_items, k, evaluate_ranking_fn):
        self.eval_users = eval_users
        self.eval_items = eval_items
        self.k = k
        self.evaluate_ranking = evaluate_ranking_fn
        self.history = []
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        
        hr, ndcg = self.evaluate_ranking(pl_module, self.eval_users, self.eval_items, self.k)
        pl_module.log(f'val_hr{self.k}', hr, prog_bar=True, logger=False, on_step=False, on_epoch=True)
        pl_module.train()
        
        m = trainer.callback_metrics
        self.history.append({
            'epoch': trainer.current_epoch,
            'train_loss': float(m.get('train_loss', float('nan'))),
            'val_loss': float(m.get('val_loss', float('nan'))),
            'train_auc': float(m.get('train_auc', float('nan'))),
            'val_auc': float(m.get('val_auc', float('nan'))),
            f'val_hr@{self.k}': hr,
            f'val_ndcg@{self.k}': ndcg,
            'lr': trainer.optimizers[0].param_groups[0]['lr'],
        })
        
        print(f"\n  [Epoch {trainer.current_epoch}] HR@{self.k}={hr:.4f} | NDCG@{self.k}={ndcg:.4f}")
    
    def save(self, result_path):
        hist_df = pd.DataFrame(self.history)
        csv_path = os.path.join(result_path, 'training_history.csv')
        hist_df.to_csv(csv_path, index=False)
        
        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss plot
        axes[0][0].plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss')
        axes[0][0].plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss')
        axes[0][0].set_xlabel('Epoch')
        axes[0][0].set_ylabel('MSE Loss')
        axes[0][0].set_title('Loss')
        axes[0][0].legend()
        axes[0][0].grid(alpha=0.3)
        
        # AUC plot
        axes[0][1].plot(hist_df['epoch'], hist_df['train_auc'], label='Train AUC')
        axes[0][1].plot(hist_df['epoch'], hist_df['val_auc'], label='Val AUC')
        axes[0][1].set_xlabel('Epoch')
        axes[0][1].set_ylabel('ROC AUC')
        axes[0][1].set_title('ROC AUC')
        axes[0][1].legend()
        axes[0][1].grid(alpha=0.3)
        
        # Ranking metrics plot
        axes[1][0].plot(hist_df['epoch'], hist_df[f'val_hr@{self.k}'], label=f'HR@{self.k}', marker='o', ms=3)
        axes[1][0].plot(hist_df['epoch'], hist_df[f'val_ndcg@{self.k}'], label=f'NDCG@{self.k}', marker='s', ms=3)
        axes[1][0].set_xlabel('Epoch')
        axes[1][0].set_ylabel('Score')
        axes[1][0].set_title(f'Ranking Metrics')
        axes[1][0].legend()
        axes[1][0].grid(alpha=0.3)
        
        # Learning rate plot
        axes[1][1].plot(hist_df['epoch'], hist_df['lr'], color='tab:red')
        axes[1][1].set_xlabel('Epoch')
        axes[1][1].set_ylabel('Learning Rate')
        axes[1][1].set_title('LR Schedule (Adam + Warmup + Cosine)')
        axes[1][1].grid(alpha=0.3)
        
        plt.suptitle('Training History — NeuMF with MSE Loss', fontsize=14)
        plt.tight_layout()
        curves_path = os.path.join(result_path, 'learning_curves.png')
        plt.savefig(curves_path, dpi=150)
        plt.close()
        
        return csv_path, curves_path
