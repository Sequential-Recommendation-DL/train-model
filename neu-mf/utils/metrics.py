# utils/metrics.py
import torch
import numpy as np

def evaluate_ranking(model, users_mat, items_mat, k, batch_size=512):
    """Single Responsibility: Ranking evaluation (HR@K, NDCG@K)"""
    model.eval()
    device = next(model.parameters()).device
    ranks = []
    
    for i in range(0, len(users_mat), batch_size):
        u = users_mat[i:i+batch_size].to(device)
        it = items_mat[i:i+batch_size].to(device)
        scores = model(u.reshape(-1), it.reshape(-1)).reshape(u.shape)
        rank = (scores > scores[:, :1]).sum(dim=1) + 1
        ranks.append(rank.cpu())
    
    ranks = torch.cat(ranks).float()
    hr = (ranks <= k).float().mean().item()
    ndcg = torch.where(ranks <= k, 1.0 / torch.log2(ranks + 1), torch.zeros_like(ranks)).mean().item()
    
    return hr, ndcg
