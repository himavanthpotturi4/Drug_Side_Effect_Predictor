from __future__ import annotations

from torch import nn
import torch


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.linear(adj @ x)


class ArchitectureGNN(nn.Module):
    def __init__(self, num_nodes: int, num_drugs: int, num_side_effects: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.num_drugs = num_drugs
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.side_effect_head = nn.Linear(hidden_dim, num_side_effects)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def encode(self, adj: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        x = self.dropout(self.relu(self.gcn1(x, adj)))
        x = self.dropout(self.relu(self.gcn2(x, adj)))
        return x

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encode(adj)
        drug_embeddings = embeddings[: self.num_drugs]
        return (
            self.side_effect_head(drug_embeddings),
            self.action_head(drug_embeddings),
            embeddings,
        )

