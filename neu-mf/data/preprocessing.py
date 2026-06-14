# data/preprocessing.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class DataPreprocessor:
    """Single Responsibility: Data loading and preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.user2idx = {}
        self.item2idx = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw data"""
        train_df = pd.read_csv(f"{self.config.DATA_PATH}train.csv")
        val_df = pd.read_csv(f"{self.config.DATA_PATH}val.csv")
        
        # Rename columns
        train_df = train_df.rename(columns={
            "UserId": "user_id", "ItemId": "item_id",
            "Timestamp": "timestamp", "Label": "label"
        })
        val_df = val_df.rename(columns={
            "UserId": "user_id", "ItemId": "item_id",
            "Timestamp": "timestamp", "Label": "label"
        })

        print(f"✓ Train data: {train_df.shape[0]:,} samples")
        print(f"✓ Validation data: {val_df.shape[0]:,} samples")
        
        return train_df, val_df
    
    def build_mappings(self, train_df: pd.DataFrame) -> None:
        """Build ID mappings from training data only"""
        all_user_ids = train_df['user_id'].unique()
        all_item_ids = train_df['item_id'].unique()
        
        self.user2idx = {u: i for i, u in enumerate(all_user_ids)}
        self.item2idx = {m: i for i, m in enumerate(all_item_ids)}
        
    def transform_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform data to indices"""
        train_df['user_idx'] = train_df['user_id'].map(self.user2idx)
        train_df['item_idx'] = train_df['item_id'].map(self.item2idx)
        val_df['user_idx'] = val_df['user_id'].map(self.user2idx)
        val_df['item_idx'] = val_df['item_id'].map(self.item2idx)
        
        # Remove unmapped validation rows
        n_unmapped = val_df[['user_idx', 'item_idx']].isna().any(axis=1).sum()
        if n_unmapped > 0:
            print(f"⚠ Removed {n_unmapped:,} cold-start validation rows")
            val_df = val_df.dropna(subset=['user_idx', 'item_idx'])
        
        val_df['user_idx'] = val_df['user_idx'].astype(int)
        val_df['item_idx'] = val_df['item_idx'].astype(int)
        
        return train_df, val_df
    
    def build_popularity_distribution(self, train_df: pd.DataFrame, num_items: int) -> np.ndarray:
        """Build popularity-based sampling distribution"""
        pop_counts = train_df.groupby('item_idx').size().reindex(range(num_items), fill_value=0).values
        pop_probs = np.power(pop_counts, self.config.POP_ALPHA)
        pop_probs = pop_probs / pop_probs.sum()
        return np.cumsum(pop_probs)
    
    def get_interacted_items(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> Dict:
        """Get interacted items for each user"""
        if val_df is not None:
            combined = pd.concat([train_df, val_df])
            return combined.groupby('user_idx')['item_idx'].apply(set).to_dict()
        return train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
