from __future__ import annotations

from difflib import get_close_matches

import pandas as pd
import torch

from .config import MODELS_DIR, PROCESSED_DIR
from .graph_data import load_graph_batch
from .model import DrugGNN
from .utils import load_json, normalize_text


class Predictor:
    def __init__(self):
        checkpoint = torch.load(MODELS_DIR / "drug_gnn.pt", map_location="cpu", weights_only=True)
        self.metadata = load_json(MODELS_DIR / "metadata.json")
        self.threshold = float(self.metadata["decision_threshold"])

        self.drug_index = pd.read_csv(PROCESSED_DIR / "drug_index.csv")
        self.side_index = pd.read_csv(PROCESSED_DIR / "side_effect_index.csv")
        self.action_index = pd.read_csv(PROCESSED_DIR / "action_index.csv")
        self.batch = load_graph_batch()

        self.model = DrugGNN(
            num_nodes=checkpoint["num_nodes"],
            num_drugs=checkpoint["num_drugs"],
            num_side_effects=checkpoint["num_side_effects"],
            num_actions=checkpoint["num_actions"],
            hidden_dim=checkpoint["config"]["hidden_dim"],
            num_layers=checkpoint["config"]["num_layers"],
            dropout=checkpoint["config"]["dropout"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self._name_to_idx = dict(zip(self.drug_index["drug_name"], self.drug_index["drug_idx"]))
        self._idx_to_name = dict(zip(self.drug_index["drug_idx"], self.drug_index["drug_name"]))

    def resolve_drug(self, drug_name: str) -> tuple[str, int] | tuple[None, None]:
        norm = normalize_text(drug_name)
        if norm in self._name_to_idx:
            return norm, int(self._name_to_idx[norm])
        matches = get_close_matches(norm, self._name_to_idx.keys(), n=1, cutoff=0.82)
        if not matches:
            return None, None
        best = matches[0]
        return best, int(self._name_to_idx[best])

    def predict(self, drug_name: str, top_k: int = 12) -> dict:
        resolved_name, drug_idx = self.resolve_drug(drug_name)
        if resolved_name is None:
            return {
                "found": False,
                "message": "Drug not found in trained dataset vocabulary.",
            }

        with torch.no_grad():
            side_logits, action_logits = self.model(self.batch.adjacency)
            side_prob = torch.sigmoid(side_logits[drug_idx]).cpu().numpy()
            action_prob = torch.softmax(action_logits[drug_idx], dim=0).cpu().numpy()

        top_idx = side_prob.argsort()[::-1][:top_k]
        predicted_side_effects = [
            {
                "side_effect": self.side_index.iloc[i]["side_effect_name"],
                "probability": float(side_prob[i]),
                "positive_at_threshold": bool(side_prob[i] >= self.threshold),
            }
            for i in top_idx
        ]

        action_i = int(action_prob.argmax())
        predicted_action = {
            "class": self.action_index.iloc[action_i]["action_class"],
            "confidence": float(action_prob[action_i]),
            "all_probabilities": {
                row["action_class"]: float(action_prob[idx])
                for idx, row in self.action_index.reset_index(drop=True).iterrows()
            },
        }
        return {
            "found": True,
            "input_drug": drug_name,
            "resolved_drug": resolved_name,
            "drug_idx": drug_idx,
            "predicted_action": predicted_action,
            "predicted_side_effects": predicted_side_effects,
        }
