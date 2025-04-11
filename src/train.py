import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

class Trainer:
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, float, float]:
        self.model.eval()
        predictions: List[int] = []
        true_labels: List[int] = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        return accuracy, precision, recall, f1

    def train(self, train_loader: DataLoader, test_loader: DataLoader, 
             num_epochs: int = 5, learning_rate: float = 2e-5) -> Dict[str, List[float]]:
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        best_f1 = 0.0
        history: Dict[str, List[float]] = {
            'train_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            accuracy, precision, recall, f1 = self.evaluate(test_loader)

            history['train_loss'].append(train_loss)
            history['accuracy'].append(accuracy)
            history['precision'].append(precision)
            history['recall'].append(recall)
            history['f1'].append(f1)

            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), 'best_model.pt')

        return history

    def plot_metrics(self, history: Dict[str, List[float]]) -> None:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(history['accuracy'])
        plt.title('Accuracy')
        plt.subplot(2, 2, 2)
        plt.plot(history['precision'])
        plt.title('Precision')
        plt.subplot(2, 2, 3)
        plt.plot(history['recall'])
        plt.title('Recall')
        plt.subplot(2, 2, 4)
        plt.plot(history['f1'])
        plt.title('F1 Score')
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close() 