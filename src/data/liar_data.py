import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LiarDataset(Dataset):
    def __init__(self, statements: List[str], labels: List[int], tokenizer: AutoTokenizer,
                 max_length: int = 512) -> None:
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.statements)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        statement = str(self.statements[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            statement,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class LiarContrastiveDataModule:
    def __init__(self, data_path: str, tokenizer_name: str = 'bert-base-uncased',
                 batch_size: int = 32, max_length: int = 512, val_split: float = 0.1,
                 test_split: float = 0.1, random_state: int = 42) -> None:
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state

    def prepare_data(self) -> None:
        df = pd.read_csv(self.data_path, sep='\t')
        statements = df['statement'].tolist()
        labels = (df['label'] == 'false').astype(int).tolist()

        train_statements, temp_statements, train_labels, temp_labels = train_test_split(
            statements, labels, test_size=self.val_split + self.test_split,
            random_state=self.random_state, stratify=labels
        )

        val_statements, test_statements, val_labels, test_labels = train_test_split(
            temp_statements, temp_labels, test_size=self.test_split / (self.val_split + self.test_split),
            random_state=self.random_state, stratify=temp_labels
        )

        self.train_dataset = LiarDataset(
            train_statements, train_labels, self.tokenizer, self.max_length
        )
        self.val_dataset = LiarDataset(
            val_statements, val_labels, self.tokenizer, self.max_length
        )
        self.test_dataset = LiarDataset(
            test_statements, test_labels, self.tokenizer, self.max_length
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        ) 