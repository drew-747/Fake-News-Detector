import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1,
                 alpha: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class GraphModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', num_classes: int = 2,
                 feature_dim: int = 7, hidden_dim: int = 256, dropout: float = 0.1,
                 n_heads: int = 8) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(self.bert.config.hidden_size, hidden_dim, dropout=dropout)
            for _ in range(n_heads)
        ])
        
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim - 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.language_embedding = nn.Embedding(7, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_heads + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def build_adjacency_matrix(self, texts: List[str], threshold: float = 0.5) -> torch.Tensor:
        embeddings = []
        for text in texts:
            with torch.no_grad():
                encoding = self.bert(text, return_tensors='pt')
                embedding = self.bert(**encoding).last_hidden_state[:, 0, :]
                embeddings.append(embedding)
        
        embeddings = torch.cat(embeddings, dim=0)
        similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        adj = (similarity > threshold).float()
        return adj

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = input_ids.size(0)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # Build graph for the batch
        texts = [self.bert.config.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids]
        adj = self.build_adjacency_matrix(texts)
        
        # Graph attention
        x = sequence_output[:, 0, :]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Feature processing
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            combined = torch.cat([x, processed_features, language_emb], dim=1)
        else:
            combined = x
            
        logits = self.classifier(combined)
        return logits

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      features: Optional[torch.Tensor] = None) -> torch.Tensor:
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        texts = [self.bert.config.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids]
        adj = self.build_adjacency_matrix(texts)
        
        x = sequence_output[:, 0, :]
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        
        if features is not None:
            processed_features = self.feature_processor(features[:, :-1])
            language_emb = self.language_embedding(features[:, -1].long())
            return torch.cat([x, processed_features, language_emb], dim=1)
        return x 