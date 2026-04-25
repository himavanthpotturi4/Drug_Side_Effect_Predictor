from __future__ import annotations


def run_module_9_output_visualization(
    *,
    input_drug: str,
    resolved_drug: str,
    predicted_action: dict,
    predicted_side_effects: list[dict],
) -> dict:
    return {
        "input_drug": input_drug,
        "resolved_drug": resolved_drug,
        "predicted_action": predicted_action,
        "predicted_side_effects": predicted_side_effects,
        "summary": {
            "predicted_target_class": predicted_action["class"],
            "target_confidence": predicted_action["confidence"],
            "top_side_effect": predicted_side_effects[0]["side_effect"] if predicted_side_effects else None,
        },
    }

