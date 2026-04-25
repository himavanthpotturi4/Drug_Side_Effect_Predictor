from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .config import PROCESSED_DIR


@dataclass
class GraphBatch:
    adjacency: torch.Tensor
    drug_indices: torch.Tensor
    side_effect_labels: torch.Tensor
    action_labels: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_drugs: int
    num_side_effects: int
    num_actions: int


def _symmetric_norm_adj(num_nodes: int, rows: np.ndarray, cols: np.ndarray) -> torch.Tensor:
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adj = adj + sp.eye(num_nodes, dtype=np.float32, format="coo")
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5, where=degree > 0).astype(np.float32)
    d_mat = sp.diags(d_inv_sqrt)
    norm = d_mat @ adj @ d_mat
    return torch.tensor(norm.toarray(), dtype=torch.float32)


def load_graph_batch() -> GraphBatch:
    drug_index = pd.read_csv(PROCESSED_DIR / "drug_index.csv")
    side_index = pd.read_csv(PROCESSED_DIR / "side_effect_index.csv")
    action_index = pd.read_csv(PROCESSED_DIR / "action_index.csv")
    edges = pd.read_csv(PROCESSED_DIR / "drug_side_effect_edges.csv")
    action_labels = pd.read_csv(PROCESSED_DIR / "drug_action_labels.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits.csv")

    num_drugs = len(drug_index)
    num_side = len(side_index)
    num_actions = len(action_index)
    num_nodes = num_drugs + num_side

    drug_nodes = edges["drug_idx"].to_numpy()
    side_nodes = edges["side_effect_idx"].to_numpy() + num_drugs
    rows = np.concatenate([drug_nodes, side_nodes])
    cols = np.concatenate([side_nodes, drug_nodes])
    adjacency = _symmetric_norm_adj(num_nodes, rows, cols)

    y_side = np.zeros((num_drugs, num_side), dtype=np.float32)
    y_side[edges["drug_idx"].to_numpy(), edges["side_effect_idx"].to_numpy()] = 1.0

    y_action = np.zeros(num_drugs, dtype=np.int64)
    y_action[action_labels["drug_idx"].to_numpy()] = action_labels["action_idx"].to_numpy()

    split_by_drug = splits.set_index("drug_idx")["split"].to_dict()
    train_mask = np.array([split_by_drug.get(i) == "train" for i in range(num_drugs)], dtype=bool)
    val_mask = np.array([split_by_drug.get(i) == "val" for i in range(num_drugs)], dtype=bool)
    test_mask = np.array([split_by_drug.get(i) == "test" for i in range(num_drugs)], dtype=bool)

    return GraphBatch(
        adjacency=adjacency,
        drug_indices=torch.arange(num_drugs, dtype=torch.long),
        side_effect_labels=torch.tensor(y_side, dtype=torch.float32),
        action_labels=torch.tensor(y_action, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        num_drugs=num_drugs,
        num_side_effects=num_side,
        num_actions=num_actions,
    )
