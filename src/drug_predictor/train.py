from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR
from .graph_data import load_graph_batch
from .model import DrugGNN
from .utils import ensure_dir, save_json


@dataclass
class TrainConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 70
    side_effect_loss_weight: float = 1.0
    action_loss_weight: float = 0.5
    decision_threshold: float = 0.35


def _compute_metrics(
    side_logits: torch.Tensor,
    action_logits: torch.Tensor,
    y_side: torch.Tensor,
    y_action: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> dict:
    side_true = y_side[mask].cpu().numpy()
    side_pred = (torch.sigmoid(side_logits[mask]).cpu().numpy() >= threshold).astype(np.int32)
    action_true = y_action[mask].cpu().numpy()
    action_pred = torch.argmax(action_logits[mask], dim=1).cpu().numpy()

    metrics = {
        "side_effect_f1_micro": float(
            f1_score(side_true, side_pred, average="micro", zero_division=0)
        ),
        "side_effect_f1_macro": float(
            f1_score(side_true, side_pred, average="macro", zero_division=0)
        ),
        "action_accuracy": float(accuracy_score(action_true, action_pred)),
    }
    return metrics


def train_and_save(config: TrainConfig | None = None) -> dict:
    cfg = config or TrainConfig()
    ensure_dir(MODELS_DIR)
    ensure_dir(OUTPUTS_DIR)

    batch = load_graph_batch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DrugGNN(
        num_nodes=batch.adjacency.shape[0],
        num_drugs=batch.num_drugs,
        num_side_effects=batch.num_side_effects,
        num_actions=batch.num_actions,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    adjacency = batch.adjacency.to(device)
    y_side = batch.side_effect_labels.to(device)
    y_action = batch.action_labels.to(device)
    train_mask = batch.train_mask.to(device)
    val_mask = batch.val_mask.to(device)
    test_mask = batch.test_mask.to(device)

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        side_logits, action_logits = model(adjacency)
        loss = (
            cfg.side_effect_loss_weight * bce(side_logits[train_mask], y_side[train_mask])
            + cfg.action_loss_weight * ce(action_logits[train_mask], y_action[train_mask])
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            side_logits, action_logits = model(adjacency)
            val_metrics = _compute_metrics(
                side_logits,
                action_logits,
                y_side,
                y_action,
                val_mask,
                threshold=cfg.decision_threshold,
            )
            score = 0.7 * val_metrics["side_effect_f1_micro"] + 0.3 * val_metrics["action_accuracy"]
            history.append({"epoch": epoch, "loss": float(loss.item()), **val_metrics, "score": float(score)})
            if score > best_val:
                best_val = score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        side_logits, action_logits = model(adjacency)
        test_metrics = _compute_metrics(
            side_logits,
            action_logits,
            y_side,
            y_action,
            test_mask,
            threshold=cfg.decision_threshold,
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "num_nodes": batch.adjacency.shape[0],
            "num_drugs": batch.num_drugs,
            "num_side_effects": batch.num_side_effects,
            "num_actions": batch.num_actions,
        },
        MODELS_DIR / "drug_gnn.pt",
    )

    pd.DataFrame(history).to_csv(OUTPUTS_DIR / "training_history.csv", index=False)
    save_json(OUTPUTS_DIR / "test_metrics.json", test_metrics)

    metadata = {
        "decision_threshold": cfg.decision_threshold,
        "drug_index_path": str(PROCESSED_DIR / "drug_index.csv"),
        "side_effect_index_path": str(PROCESSED_DIR / "side_effect_index.csv"),
        "action_index_path": str(PROCESSED_DIR / "action_index.csv"),
        "metrics": test_metrics,
    }
    save_json(MODELS_DIR / "metadata.json", metadata)
    return test_metrics
