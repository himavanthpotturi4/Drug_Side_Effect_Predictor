from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import ACTION_CLASSES, PROCESSED_DIR, RAW_DIR, TOP_SIDE_EFFECTS
from ..preprocessing import load_sider_tables
from ..utils import ensure_dir, normalize_text


def run_module_2_data_preprocessing() -> dict[str, Path]:
    ensure_dir(PROCESSED_DIR)
    se_df, names_df = load_sider_tables()

    se_df["side_effect_name"] = se_df["side_effect_name"].fillna("").map(normalize_text)
    names_df["drug_name"] = names_df["drug_name"].fillna("").map(normalize_text)

    # Diagram requirements: handle missing values, remove duplicates, standardize text, assign unique indices.
    merged = se_df.merge(names_df[["stitch_flat", "drug_name"]], on="stitch_flat", how="left")
    merged = merged[["drug_name", "side_effect_name"]].dropna().drop_duplicates()
    merged = merged[(merged["drug_name"] != "") & (merged["side_effect_name"] != "")]

    top_side_effects = merged["side_effect_name"].value_counts().head(TOP_SIDE_EFFECTS).index.tolist()
    merged = merged[merged["side_effect_name"].isin(top_side_effects)]

    action_df = pd.read_csv(RAW_DIR / "drug_action_labels.csv", dtype=str)
    action_df["drug_name"] = action_df["drug_name"].fillna("").map(normalize_text)
    action_df["action_class"] = action_df["action_class"].fillna("other").map(normalize_text)
    action_df["action_class"] = action_df["action_class"].where(
        action_df["action_class"].isin(ACTION_CLASSES), "other"
    )

    disease_path = RAW_DIR / "drug_disease_labels.csv"
    if disease_path.exists() and disease_path.stat().st_size > 0:
        disease_df = pd.read_csv(disease_path, dtype=str)
        disease_df["drug_name"] = disease_df["drug_name"].fillna("").map(normalize_text)
        disease_df["disease_name"] = disease_df["disease_name"].fillna("unknown_disease").map(normalize_text)
    else:
        disease_df = pd.DataFrame(columns=["drug_name", "disease_name"])

    symptom_path = RAW_DIR / "drug_symptom_labels.csv"
    if symptom_path.exists() and symptom_path.stat().st_size > 0:
        symptom_df = pd.read_csv(symptom_path, dtype=str)
        symptom_df["drug_name"] = symptom_df["drug_name"].fillna("").map(normalize_text)
        symptom_df["symptom_name"] = symptom_df["symptom_name"].fillna("").map(normalize_text)
        symptom_df = symptom_df[(symptom_df["drug_name"] != "") & (symptom_df["symptom_name"] != "")]
        symptom_df = symptom_df.drop_duplicates()
    else:
        # Fallback only when raw symptom table is unavailable.
        symptom_df = merged.rename(columns={"side_effect_name": "symptom_name"})

    drug_index = pd.DataFrame(
        {"drug_name": sorted(merged["drug_name"].unique()), "drug_idx": range(merged["drug_name"].nunique())}
    )
    side_effect_index = pd.DataFrame(
        {"side_effect_name": sorted(merged["side_effect_name"].unique()), "side_effect_idx": range(merged["side_effect_name"].nunique())}
    )
    symptom_index = pd.DataFrame(
        {"symptom_name": sorted(symptom_df["symptom_name"].unique()), "symptom_idx": range(symptom_df["symptom_name"].nunique())}
    )
    disease_index = pd.DataFrame(
        {"disease_name": sorted(disease_df["disease_name"].dropna().unique()), "disease_idx": range(disease_df["disease_name"].dropna().nunique())}
    )
    action_index = pd.DataFrame({"action_class": ACTION_CLASSES, "action_idx": range(len(ACTION_CLASSES))})

    drug_side_effect_edges = (
        merged.merge(drug_index, on="drug_name", how="inner")
        .merge(side_effect_index, on="side_effect_name", how="inner")
        [["drug_idx", "side_effect_idx"]]
        .drop_duplicates()
    )
    drug_symptom_edges = (
        symptom_df.merge(drug_index, on="drug_name", how="inner")
        .merge(symptom_index, on="symptom_name", how="inner")
        [["drug_idx", "symptom_idx"]]
        .drop_duplicates()
    )
    drug_disease_edges = (
        disease_df.merge(drug_index, on="drug_name", how="inner")
        .merge(disease_index, on="disease_name", how="inner")
        [["drug_idx", "disease_idx"]]
        .drop_duplicates()
    )

    drug_action_labels = (
        drug_index.merge(action_df[["drug_name", "action_class"]], on="drug_name", how="left")
        .merge(action_index, on="action_class", how="left")
        .fillna({"action_idx": action_index.loc[action_index["action_class"] == "other", "action_idx"].iloc[0]})
        [["drug_idx", "action_idx"]]
        .astype({"action_idx": int})
    )

    paths = {
        "drug_index": PROCESSED_DIR / "drug_index.csv",
        "side_effect_index": PROCESSED_DIR / "side_effect_index.csv",
        "symptom_index": PROCESSED_DIR / "symptom_index.csv",
        "disease_index": PROCESSED_DIR / "disease_index.csv",
        "action_index": PROCESSED_DIR / "action_index.csv",
        "drug_side_effect_edges": PROCESSED_DIR / "drug_side_effect_edges.csv",
        "drug_symptom_edges": PROCESSED_DIR / "drug_symptom_edges.csv",
        "drug_disease_edges": PROCESSED_DIR / "drug_disease_edges.csv",
        "drug_action_labels": PROCESSED_DIR / "drug_action_labels.csv",
    }
    drug_index.to_csv(paths["drug_index"], index=False)
    side_effect_index.to_csv(paths["side_effect_index"], index=False)
    symptom_index.to_csv(paths["symptom_index"], index=False)
    disease_index.to_csv(paths["disease_index"], index=False)
    action_index.to_csv(paths["action_index"], index=False)
    drug_side_effect_edges.to_csv(paths["drug_side_effect_edges"], index=False)
    drug_symptom_edges.to_csv(paths["drug_symptom_edges"], index=False)
    drug_disease_edges.to_csv(paths["drug_disease_edges"], index=False)
    drug_action_labels.to_csv(paths["drug_action_labels"], index=False)
    return paths
