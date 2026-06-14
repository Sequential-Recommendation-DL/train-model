# data/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict

class NeuMFTrainDataset(Dataset):
    """Single Responsibility: Training data with dynamic negative sampling"""
    
    def __init__(self, pos_df, interacted_dict: Dict, num_items: int, 
                 num_neg: int, pop_cum: np.ndarray):
        self.users = pos_df['user_idx'].values
        self.items = pos_df['item_idx'].values
        self.labels = (pos_df['label'].values > 0).astype(np.float32)
        self.interacted = interacted_dict
        self.num_items = num_items
        self.num_neg = num_neg
        self.pop_cum = pop_cum

    def __len__(self):
        return len(self.users) * (1 + self.num_neg)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_neg)
        slot = idx % (1 + self.num_neg)
        u = self.users[pos_idx]
        
        if slot == 0:  # positive
            return (torch.tensor(u, dtype=torch.long),
                    torch.tensor(self.items[pos_idx], dtype=torch.long),
                    torch.tensor(self.labels[pos_idx], dtype=torch.float32))
        
        # negative sampling
        interacted = self.interacted[u]
        while True:
            j = int(np.searchsorted(self.pop_cum, np.random.rand()))
            if j not in interacted:
                return (torch.tensor(u, dtype=torch.long),
                        torch.tensor(j, dtype=torch.long),
                        torch.tensor(0.0, dtype=torch.float32))

class NeuMFDataset(Dataset):
    """Single Responsibility: Static dataset for validation"""
    
    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(dataframe['item_idx'].values, dtype=torch.long)
        self.labels = torch.tensor((dataframe['label'].values > 0).astype('float32'), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
