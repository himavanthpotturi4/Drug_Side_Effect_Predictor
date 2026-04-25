from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def run_module_6_link_prediction(
    side_effect_logits: torch.Tensor,
    side_effect_index: pd.DataFrame,
    drug_idx: int,
    top_k: int = 15,
    threshold: float = 0.35,
) -> list[dict]:
    probs = torch.sigmoid(side_effect_logits[drug_idx]).detach().cpu().numpy()
    order = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "side_effect": side_effect_index.iloc[i]["side_effect_name"],
            "probability": float(probs[i]),
            "positive_at_threshold": bool(probs[i] >= threshold),
        }
        for i in order
    ]

