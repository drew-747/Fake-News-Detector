import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import optuna
from torch_geometric.nn import GATConv
from sentence_transformers import SentenceTransformer
import spacy

class AdvancedModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', num_classes: int = 2,
                 feature_dim: int = 7, hidden_dim: int = 256, dropout: float = 0.1,
                 num_heads: int = 8, num_layers: int = 2, num_filters: int = 100,
                 filter_sizes: List[int] = [3, 4, 5]) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # CNN for local patterns
        self.convs = nn.ModuleList([
            nn.Conv1d(self.bert.config.hidden_size, num_filters, k) for k in filter_sizes
        ])
        
        # LSTM for sequential patterns
        self.lstm = nn.LSTM(
            input_size=num_filters * len(filter_sizes),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # Transformer for global context
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.bert.config.hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Graph attention for relationships
        self.gat = GATConv(
            in_channels=self.bert.config.hidden_size,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout
        )
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim - 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Language embeddings
        self.language_embedding = nn.Embedding(7, hidden_dim)
        
        # Attention mechanisms
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Additional NLP tools
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')

    def extract_linguistic_features(self, text: str) -> torch.Tensor:
        doc = self.nlp(text)
        features = []
        
        # Extract linguistic features
        features.append(len(doc))  # Document length
        features.append(len([token for token in doc if token.is_alpha]))  # Word count
        features.append(len([token for token in doc if token.is_stop]))  # Stop word count
        features.append(len([ent for ent in doc.ents]))  # Named entity count
        features.append(len([token for token in doc if token.pos_ == 'VERB']))  # Verb count
        features.append(len([token for token in doc if token.pos_ == 'NOUN']))  # Noun count
        features.append(len([token for token in doc if token.pos_ == 'ADJ']))  # Adjective count
        
        return torch.tensor(features, dtype=torch.float)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = input_ids.size(0)
        
        # BERT encoding
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # CNN processing
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(sequence_output.transpose(1, 2))
            conv_out = F.relu(conv_out)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        cnn_output = torch.cat(conv_outputs, 1)
        
        # LSTM processing
        lstm_output, _ = self.lstm(cnn_output.unsqueeze(1))
        
        # Transformer processing
        transformer_output = self.transformer(sequence_output)
        
        # Self-attention
        attn_output, _ = self.self_attention(
            transformer_output,
            transformer_output,
            transformer_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Graph processing
        texts = [self.bert.config.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids]
        sentence_embeddings = self.sentence_transformer.encode(texts)
        edge_index = self.build_graph_edges(sentence_embeddings)
        graph_output = self.gat(sequence_output[:, 0, :], edge_index)
        
        # Feature processing
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            
            # Cross-attention between features and text
            cross_attn, _ = self.cross_attention(
                processed_features.unsqueeze(1),
                attn_output,
                attn_output,
                key_padding_mask=~attention_mask.bool()
            )
            cross_attn = cross_attn.squeeze(1)
            
            combined = torch.cat([
                torch.mean(attn_output, dim=1),
                torch.mean(lstm_output, dim=1),
                graph_output,
                processed_features,
                language_emb,
                cross_attn
            ], dim=1)
        else:
            combined = torch.cat([
                torch.mean(attn_output, dim=1),
                torch.mean(lstm_output, dim=1),
                graph_output
            ], dim=1)
        
        logits = self.classifier(combined)
        return logits

    def build_graph_edges(self, embeddings: np.ndarray, threshold: float = 0.5) -> torch.Tensor:
        similarity = np.dot(embeddings, embeddings.T)
        edges = np.where(similarity > threshold)
        return torch.tensor(edges, dtype=torch.long)

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      features: Optional[torch.Tensor] = None) -> torch.Tensor:
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # Process through all components
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(sequence_output.transpose(1, 2))
            conv_out = F.relu(conv_out)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        cnn_output = torch.cat(conv_outputs, 1)
        
        lstm_output, _ = self.lstm(cnn_output.unsqueeze(1))
        transformer_output = self.transformer(sequence_output)
        
        texts = [self.bert.config.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids]
        sentence_embeddings = self.sentence_transformer.encode(texts)
        edge_index = self.build_graph_edges(sentence_embeddings)
        graph_output = self.gat(sequence_output[:, 0, :], edge_index)
        
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            return torch.cat([
                torch.mean(transformer_output, dim=1),
                torch.mean(lstm_output, dim=1),
                graph_output,
                processed_features,
                language_emb
            ], dim=1)
        return torch.cat([
            torch.mean(transformer_output, dim=1),
            torch.mean(lstm_output, dim=1),
            graph_output
        ], dim=1) 