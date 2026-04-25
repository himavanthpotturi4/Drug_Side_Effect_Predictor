from __future__ import annotations

import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.linear(adj @ x)


class DrugGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_drugs: int,
        num_side_effects: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_drugs = num_drugs
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.gcn_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.side_effect_head = nn.Linear(hidden_dim, num_side_effects)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def encode(self, adj: torch.Tensor) -> torch.Tensor:
        x = self.node_embeddings.weight
        for layer in self.gcn_layers:
            x = layer(x, adj)
            x = self.activation(x)
            x = self.dropout(x)
        return x

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encode(adj)
        drug_x = x[: self.num_drugs]
        side_logits = self.side_effect_head(drug_x)
        action_logits = self.action_head(drug_x)
        return side_logits, action_logits
