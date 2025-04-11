import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Any, Optional, List

class HybridModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', num_classes: int = 2,
                 feature_dim: int = 7, hidden_dim: int = 256, dropout: float = 0.1,
                 num_filters: int = 100, filter_sizes: List[int] = [3, 4, 5]) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(self.bert.config.hidden_size, num_filters, k) for k in filter_sizes
        ])
        
        self.lstm = nn.LSTM(
            input_size=num_filters * len(filter_sizes),
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim - 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.language_embedding = nn.Embedding(7, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # CNN
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(sequence_output.transpose(1, 2))
            conv_out = torch.relu(conv_out)
            conv_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        cnn_output = torch.cat(conv_outputs, 1)
        
        # LSTM
        lstm_output, _ = self.lstm(cnn_output.unsqueeze(1))
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        
        # Feature processing
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            combined = torch.cat([attended_output, processed_features, language_emb], dim=1)
        else:
            combined = attended_output
            
        logits = self.classifier(combined)
        return logits

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      features: Optional[torch.Tensor] = None) -> torch.Tensor:
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(sequence_output.transpose(1, 2))
            conv_out = torch.relu(conv_out)
            conv_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        cnn_output = torch.cat(conv_outputs, 1)
        
        lstm_output, _ = self.lstm(cnn_output.unsqueeze(1))
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            return torch.cat([attended_output, processed_features, language_emb], dim=1)
        return attended_output 