import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from typing import List, Union, Any, Optional
import numpy as np

class BERTClassifier(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class EnhancedClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int = 2) -> None:
        super(EnhancedClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class TFIDFModel:
    def __init__(self, model_type: str = 'svm') -> None:
        self.vectorizer = TfidfVectorizer(max_features=5000)
        if model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100)
        elif model_type == 'lr':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(n_estimators=100)
        elif model_type == 'mlp':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        else:
            raise ValueError("Invalid model type")

    def fit(self, X: List[str], y: List[int]) -> None:
        X_transformed = self.vectorizer.fit_transform(X)
        self.model.fit(X_transformed, y)

    def predict(self, X: List[str]) -> List[int]:
        X_transformed = self.vectorizer.transform(X)
        return self.model.predict(X_transformed)

    def predict_proba(self, X: List[str]) -> List[List[float]]:
        X_transformed = self.vectorizer.transform(X)
        return self.model.predict_proba(X_transformed)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 output_dim: int, n_layers: int, dropout: float) -> None:
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

class EnsembleModel:
    def __init__(self, models: List[Any]) -> None:
        self.models = models

    def fit(self, X: List[str], y: List[int]) -> None:
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: List[str]) -> List[int]:
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.round(np.mean(predictions, axis=0)).astype(int)

    def predict_proba(self, X: List[str]) -> List[List[float]]:
        probabilities = []
        for model in self.models:
            prob = model.predict_proba(X)
            probabilities.append(prob)
        return np.mean(probabilities, axis=0) 