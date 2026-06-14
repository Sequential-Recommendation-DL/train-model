# trainers/trainer.py
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import time

class ModelTrainer:
    """Single Responsibility: Model training orchestration"""
    
    def __init__(self, config, model, train_dataloader, val_dataloader, 
                 metrics_history, result_path):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metrics_history = metrics_history
        self.result_path = result_path
        
        self.early_stopping = EarlyStopping(
            monitor='val_hr10',
            patience=config.ES_PATIENCE,
            min_delta=config.ES_MIN_DELTA,
            mode='max',
            verbose=True
        )

        self.checkpoint_callback = ModelCheckpoint(
            monitor='val_hr10',
            mode='max',
            save_top_k=1,
            dirpath=os.path.join(result_path, 'checkpoints'),
            filename='neumf-best-{epoch:02d}-{val_hr10:.4f}',
            verbose=True
        )
        
        # TensorBoard logger
        self.logger = TensorBoardLogger(
            save_dir=result_path,
            name="tensorboard_logs",
            version=f"lr{config.LEARNING_RATE}_emb{config.EMBEDDING_DIM}"
        )
    
    def train(self):
        """Train the model"""
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=self.config.MAX_EPOCHS,
            callbacks=[self.early_stopping, self.checkpoint_callback, self.metrics_history],
            logger=self.logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            default_root_dir=self.result_path
        )
        
        print("\n" + "="*60)
        print("STARTING TRAINING...")
        print(f"Optimizer: AdamW | LR: {self.config.LEARNING_RATE} | "
              f"Scheduler: Cosine with {self.config.WARMUP_EPOCHS}-epoch warmup")
        print("="*60)
        
        start_time = time.time()
        trainer.fit(self.model, self.train_dataloader, self.val_dataloader)
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print(f"TRAINING COMPLETED! Time: {training_time/60:.2f} minutes")
        print("="*60)
        
        return trainer, training_time
