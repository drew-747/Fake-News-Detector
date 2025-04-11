import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

class ContrastiveTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3)
        self.writer = SummaryWriter(f'runs/fakenews_detection/{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    def info_nce_loss(self, features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        features = nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask = mask - torch.eye(mask.shape[0], device=mask.device)
        exp_sim = torch.exp(similarity_matrix / temperature)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            self.optimizer.zero_grad()
            features = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.info_nce_loss(features, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                features = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.info_nce_loss(features, labels)
                val_loss += loss.item()

        return {'loss': val_loss / len(self.val_loader)}

    def test(self) -> Dict[str, float]:
        self.model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                features = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.info_nce_loss(features, labels)
                test_loss += loss.item()

        return {'loss': test_loss / len(self.test_loader)}

    def train(self, num_epochs: int = 10, save_dir: str = 'models') -> None:
        best_val_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)

            self.scheduler.step(val_metrics['loss'])

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_contrastive_model.pt'))

            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f}')

        test_metrics = self.test()
        print(f'Test Loss: {test_metrics["loss"]:.4f}')
        self.writer.close() 