from __future__ import annotations

import pandas as pd
import torch


def run_module_7_drug_action_classification(
    action_logits: torch.Tensor,
    action_index: pd.DataFrame,
    drug_idx: int,
) -> dict:
    probs = torch.softmax(action_logits[drug_idx], dim=0).detach().cpu().numpy()
    valid_classes = {"human_cell", "virus", "bacteria", "fungus"}
    valid_rows = [
        (i, row["action_class"])
        for i, row in action_index.reset_index(drop=True).iterrows()
        if row["action_class"] in valid_classes
    ]
    if not valid_rows:
        top = int(probs.argmax())
    else:
        top = max(valid_rows, key=lambda x: probs[x[0]])[0]
    return {
        "class": action_index.iloc[top]["action_class"],
        "confidence": float(probs[top]),
        "all_probabilities": {
            row["action_class"]: float(probs[i])
            for i, row in action_index.reset_index(drop=True).iterrows()
        },
    }
