import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTester:
    def __init__(self, model: nn.Module, test_loader: DataLoader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def test(self) -> Dict[str, Any]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs)
        }

    def evaluate(self, results: Dict[str, Any], save_dir: str = 'results') -> None:
        predictions = results['predictions']
        labels = results['labels']
        probabilities = results['probabilities']

        report = classification_report(labels, predictions, output_dict=True)
        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{save_dir}/confusion_matrix.png')
        plt.close()

        metrics = ['precision', 'recall', 'f1-score']
        plt.figure(figsize=(12, 4))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            values = [report['0'][metric], report['1'][metric]]
            plt.bar(['Real', 'Fake'], values)
            plt.title(f'{metric.capitalize()} by Class')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics.png')
        plt.close()

        print("Classification Report:")
        print(classification_report(labels, predictions))
        print("\nConfusion Matrix:")
        print(cm)

    def plot_probability_distribution(self, results: Dict[str, Any], save_dir: str = 'results') -> None:
        probabilities = results['probabilities']
        labels = results['labels']

        plt.figure(figsize=(10, 6))
        for label in [0, 1]:
            mask = labels == label
            plt.hist(probabilities[mask, 1], bins=50, alpha=0.5, label=f'Class {label}')
        plt.title('Probability Distribution')
        plt.xlabel('Probability of Fake News')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f'{save_dir}/probability_distribution.png')
        plt.close()

    def run(self, save_dir: str = 'results') -> None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        results = self.test()
        self.evaluate(results, save_dir)
        self.plot_probability_distribution(results, save_dir) 