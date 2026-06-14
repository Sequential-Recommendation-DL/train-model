# evaluation/evaluator.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, recall_score,
                             precision_score, precision_recall_curve,
                             average_precision_score)
import os

class Evaluator:
    """Single Responsibility: Final model evaluation"""
    
    def __init__(self, model, val_dataloader, result_path):
        self.model = model
        self.val_dataloader = val_dataloader
        self.result_path = result_path
    
    @torch.no_grad()
    def get_predictions(self):
        """Get all predictions and labels"""
        device = next(self.model.parameters()).device
        self.model.eval()
        
        all_preds, all_labels = [], []
        for users, items, labels in self.val_dataloader:
            users, items = users.to(device), items.to(device)
            preds = self.model(users, items).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def plot_roc_curve(self, labels, preds):
        """Plot ROC curve"""
        binary_labels = (labels > 0).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, preds)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve — Validation Set')
        plt.legend()
        plt.tight_layout()
        
        path = os.path.join(self.result_path, 'roc_curve.png')
        plt.savefig(path, dpi=150)
        plt.close()
        return path, roc_auc
    
    def find_optimal_threshold(self, labels, preds):
        """Find optimal threshold by F1 score"""
        binary_labels = (labels > 0).astype(int)
        prec, rec, thr = precision_recall_curve(binary_labels, preds)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = np.nanargmax(f1s[:-1])
        return thr[best_idx]
    
    def plot_confusion_matrix(self, labels, preds, threshold):
        """Plot confusion matrix"""
        binary_labels = (labels > 0).astype(int)
        pred_binary = (preds >= threshold).astype(int)
        cm = confusion_matrix(binary_labels, pred_binary)
        
        _, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(ax=ax, colorbar=False)
        plt.title(f'Confusion Matrix — Val (threshold={threshold:.3f})')
        plt.tight_layout()
        
        path = os.path.join(self.result_path, 'confusion_matrix.png')
        plt.savefig(path, dpi=150)
        plt.close()
        return path
    
    def plot_precision_recall_curve(self, labels, preds):
        """Plot Precision-Recall curve with AP score"""
        binary_labels = (labels > 0).astype(int)
        prec, rec, _ = precision_recall_curve(binary_labels, preds)
        ap = average_precision_score(binary_labels, preds)
        baseline = binary_labels.mean()

        plt.figure(figsize=(8, 6))
        plt.plot(rec, prec, lw=2, label=f'PR curve (AP = {ap:.4f})')
        plt.axhline(baseline, color='gray', linestyle='--', lw=1, label=f'Random baseline ({baseline:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve — Validation Set')
        plt.legend()
        plt.tight_layout()

        path = os.path.join(self.result_path, 'pr_curve.png')
        plt.savefig(path, dpi=150)
        plt.close()
        return path, ap

    def plot_score_distribution(self, labels, preds):
        """Plot predicted score distribution for positives vs negatives"""
        binary_labels = (labels > 0).astype(int)
        pos_scores = preds[binary_labels == 1]
        neg_scores = preds[binary_labels == 0]

        plt.figure(figsize=(8, 5))
        plt.hist(neg_scores, bins=60, alpha=0.6, label=f'Negative (n={len(neg_scores):,})',
                 color='steelblue', density=True)
        plt.hist(pos_scores, bins=60, alpha=0.6, label=f'Positive (n={len(pos_scores):,})',
                 color='tomato', density=True)
        plt.axvline(pos_scores.mean(), color='tomato', linestyle='--', lw=1.5,
                    label=f'Pos mean={pos_scores.mean():.3f}')
        plt.axvline(neg_scores.mean(), color='steelblue', linestyle='--', lw=1.5,
                    label=f'Neg mean={neg_scores.mean():.3f}')
        plt.xlabel('Predicted Score')
        plt.ylabel('Density')
        plt.title('Score Distribution — Positive vs Negative Samples')
        plt.legend()
        plt.tight_layout()

        path = os.path.join(self.result_path, 'score_distribution.png')
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def evaluate(self):
        """Run complete evaluation"""
        print("\nGenerating evaluation plots...")
        preds, labels = self.get_predictions()

        binary_labels = (labels > 0).astype(int)

        # ROC curve
        roc_path, roc_auc = self.plot_roc_curve(labels, preds)
        print(f"✓ ROC curve: {roc_path} (AUC = {roc_auc:.4f})")

        # Precision-Recall curve
        pr_path, pr_auc = self.plot_precision_recall_curve(labels, preds)
        print(f"✓ PR curve: {pr_path} (AP = {pr_auc:.4f})")

        # Score distribution
        dist_path = self.plot_score_distribution(labels, preds)
        print(f"✓ Score distribution: {dist_path}")

        # Optimal threshold
        best_thr = self.find_optimal_threshold(labels, preds)

        # Metrics at best threshold
        pred_best = (preds >= best_thr).astype(int)
        f1_best = f1_score(binary_labels, pred_best)
        recall_best = recall_score(binary_labels, pred_best)
        precision_best = precision_score(binary_labels, pred_best)

        print(f"\nThreshold tối ưu (theo F1): {best_thr:.4f}")
        print(f"  F1: {f1_best:.4f} | Recall: {recall_best:.4f} | Precision: {precision_best:.4f}")

        # Confusion matrix
        cm_path = self.plot_confusion_matrix(labels, preds, best_thr)
        print(f"✓ Confusion matrix: {cm_path}")

        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'best_threshold': best_thr,
            'f1_best': f1_best,
            'recall_best': recall_best,
            'precision_best': precision_best
        }
