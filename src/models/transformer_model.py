import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional

class TransformerModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', num_classes: int = 2,
                 feature_dim: int = 7, hidden_dim: int = 256, dropout: float = 0.1,
                 num_heads: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim - 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.language_embedding = nn.Embedding(7, hidden_dim)
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # Transformer Encoder
        transformer_output = self.transformer_encoder(sequence_output)
        
        # Self-Attention
        attn_output, _ = self.self_attention(
            transformer_output,
            transformer_output,
            transformer_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Pooling
        pooled_output = torch.mean(attn_output, dim=1)
        pooled_output = self.dropout(pooled_output)
        
        # Feature processing
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            combined = torch.cat([pooled_output, processed_features, language_emb], dim=1)
        else:
            combined = pooled_output
            
        logits = self.classifier(combined)
        return logits

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      features: Optional[torch.Tensor] = None) -> torch.Tensor:
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        transformer_output = self.transformer_encoder(sequence_output)
        attn_output, _ = self.self_attention(
            transformer_output,
            transformer_output,
            transformer_output,
            key_padding_mask=~attention_mask.bool()
        )
        pooled_output = torch.mean(attn_output, dim=1)
        
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            return torch.cat([pooled_output, processed_features, language_emb], dim=1)
        return pooled_output 