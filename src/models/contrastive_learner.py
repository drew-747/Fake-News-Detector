import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Any, Optional

class ContrastiveLearner(nn.Module):
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', embedding_dim: int = 768,
                 projection_dim: int = 128, feature_dim: int = 7, hidden_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim - 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.language_embedding = nn.Embedding(7, hidden_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            combined = torch.cat([pooled_output, processed_features, language_emb], dim=1)
        else:
            combined = pooled_output
            
        projection = self.projection(combined)
        return projection

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      features: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            return torch.cat([pooled_output, processed_features, language_emb], dim=1)
        return pooled_output 