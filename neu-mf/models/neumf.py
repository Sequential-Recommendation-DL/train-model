# models/neumf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from torchmetrics.classification import BinaryAUROC

class NeuMF(pl.LightningModule):
    """Single Responsibility: NeuMF model architecture"""
    
    def __init__(self, config, num_users, num_items):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config

        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, config.EMBEDDING_DIM)
        self.gmf_item_embedding = nn.Embedding(num_items, config.EMBEDDING_DIM)
        
        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, config.EMBEDDING_DIM)
        self.mlp_item_embedding = nn.Embedding(num_items, config.EMBEDDING_DIM)
        
        # MLP layers
        mlp_modules = []
        input_size = 2 * config.EMBEDDING_DIM
        for dim in config.MLP_HIDDEN_DIMS:
            mlp_modules.append(nn.Linear(input_size, dim))
            mlp_modules.append(nn.BatchNorm1d(dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=config.DROPOUT_RATE))
            input_size = dim
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        # Embedding dropout to reduce memorization of sparse training pairs
        self.emb_dropout = nn.Dropout(p=config.EMB_DROPOUT)

        # Final layer
        self.final_layer = nn.Linear(config.EMBEDDING_DIM + config.MLP_HIDDEN_DIMS[-1], 1)

        # Metrics
        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        
        nn.init.kaiming_uniform_(self.final_layer.weight, a=1, nonlinearity='relu')
        nn.init.zeros_(self.final_layer.bias)
    
    def forward(self, user_indices, item_indices):
        # GMF branch
        gmf_user_emb = self.emb_dropout(self.gmf_user_embedding(user_indices))
        gmf_item_emb = self.emb_dropout(self.gmf_item_embedding(item_indices))
        gmf_vector = gmf_user_emb * gmf_item_emb

        # MLP branch
        mlp_user_emb = self.emb_dropout(self.mlp_user_embedding(user_indices))
        mlp_item_emb = self.emb_dropout(self.mlp_item_embedding(item_indices))
        mlp_input = torch.cat((mlp_user_emb, mlp_item_emb), dim=-1)
        mlp_vector = self.mlp_layers(mlp_input)
        
        # Combine
        combined = torch.cat((gmf_vector, mlp_vector), dim=-1)
        logits = self.final_layer(combined)
        return torch.sigmoid(logits.squeeze(-1))
    
    def training_step(self, batch, batch_idx):
        user_indices, item_indices, labels = batch
        predictions = self(user_indices, item_indices)
        loss = F.binary_cross_entropy(predictions, labels)
        
        self.train_auc.update(predictions, labels.int())
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_indices, item_indices, labels = batch
        predictions = self(user_indices, item_indices)
        loss = F.binary_cross_entropy(predictions, labels)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auc.update(predictions, labels.int())
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Separate embedding params for different weight decay
        emb_params, other_params = [], []
        for name, p in self.named_parameters():
            (emb_params if 'embedding' in name else other_params).append(p)
        
        # Adam optimizer with different weight decays
        optimizer = torch.optim.AdamW(
            [{'params': emb_params, 'weight_decay': self.config.EMB_WEIGHT_DECAY},
             {'params': other_params, 'weight_decay': self.config.WEIGHT_DECAY}],
            lr=self.config.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler (cosine decay with warmup)
        def lr_lambda(epoch):
            if epoch < self.config.WARMUP_EPOCHS:
                return (epoch + 1) / self.config.WARMUP_EPOCHS
            progress = (epoch - self.config.WARMUP_EPOCHS) / max(1, self.config.MAX_EPOCHS - self.config.WARMUP_EPOCHS)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
