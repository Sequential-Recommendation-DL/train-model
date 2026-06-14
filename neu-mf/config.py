# config.py
from dataclasses import dataclass
from datetime import datetime
import os

@dataclass
class Config:
    # Paths
    DATA_PATH: str = "../data/processs/"
    RUN_NAME: str = datetime.now().strftime('%d-%m-%Y_%Hh%M')
    RESULT_PATH: str = os.path.join("results", RUN_NAME)
    
    # Model hyperparameters
    EMBEDDING_DIM: int = 16
    MLP_HIDDEN_DIMS: list = None
    DROPOUT_RATE: float = 0.4
    EMB_DROPOUT: float = 0.3
    
    # Training hyperparameters
    LEARNING_RATE: float = 2e-4
    WARMUP_EPOCHS: int = 1
    WEIGHT_DECAY: float = 1e-4
    EMB_WEIGHT_DECAY: float = 2e-3
    MAX_EPOCHS: int = 50
    BATCH_SIZE: int = 256
    NUM_NEGATIVES: int = 16
    POP_ALPHA: float = 0.0
    
    # Early stopping
    ES_MIN_DELTA: float = 1e-3
    ES_PATIENCE: int = 1
    
    # Ranking evaluation
    TOP_K: int = 10
    N_EVAL_NEG: int = 99
    N_EVAL_SAMPLES: int = 2000
    
    # Random seed
    GLOBAL_SEED: int = 42
    
    def __post_init__(self):
        if self.MLP_HIDDEN_DIMS is None:
            self.MLP_HIDDEN_DIMS = [64, 32, 16]
        os.makedirs(self.RESULT_PATH, exist_ok=True)

config = Config()
