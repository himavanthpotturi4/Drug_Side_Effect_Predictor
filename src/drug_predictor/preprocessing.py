from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.errors import EmptyDataError

from .config import ACTION_CLASSES, PROCESSED_DIR, RAW_DIR, TOP_SIDE_EFFECTS
from .utils import ensure_dir, normalize_text


def load_sider_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    se_path = RAW_DIR / "meddra_all_se.tsv.gz"
    names_path = RAW_DIR / "drug_names.tsv"

    raw_se = pd.read_csv(se_path, sep="\t", header=None, dtype=str)
    if raw_se.shape[1] >= 7:
        raw_se = raw_se.iloc[:, :7]
        raw_se.columns = [
            "stitch_flat",
            "stitch_stereo",
            "umls_label",
            "umls_cui",
            "meddra_type",
            "meddra_id",
            "side_effect_name",
        ]
        se_df = raw_se
    elif raw_se.shape[1] == 6:
        raw_se.columns = [
            "stitch_flat",
            "stitch_stereo",
            "umls_cui_from_label",
            "meddra_type",
            "umls_cui_from_meddra",
            "side_effect_name",
        ]
        se_df = raw_se
    else:
        raise ValueError("Unsupported meddra_all_se.tsv schema.")
    raw_names = pd.read_csv(names_path, sep="\t", header=None, dtype=str)
    if raw_names.shape[1] >= 4:
        raw_names = raw_names.iloc[:, :4]
        raw_names.columns = ["stitch_flat", "stitch_stereo", "pubchem_id", "drug_name"]
        names_df = raw_names
    elif raw_names.shape[1] == 2:
        raw_names.columns = ["stitch_flat", "drug_name"]
        names_df = raw_names
    else:
        raise ValueError("Unsupported drug_names.tsv format.")
    return se_df, names_df


def build_training_tables() -> dict[str, Path]:
    ensure_dir(PROCESSED_DIR)
    se_df, names_df = load_sider_tables()

    se_df["side_effect_name"] = se_df["side_effect_name"].fillna("").map(normalize_text)
    names_df["drug_name"] = names_df["drug_name"].fillna("").map(normalize_text)

    merged = se_df.merge(
        names_df[["stitch_flat", "drug_name"]],
        on="stitch_flat",
        how="left",
    )
    merged["drug_name"] = merged["drug_name"].fillna("").map(normalize_text)
    merged = merged.loc[(merged["drug_name"] != "") & (merged["side_effect_name"] != "")]
    merged = merged[["drug_name", "side_effect_name"]].drop_duplicates()

    top_side_effects = (
        merged["side_effect_name"].value_counts().head(TOP_SIDE_EFFECTS).index.tolist()
    )
    merged = merged.loc[merged["side_effect_name"].isin(top_side_effects)]

    action_path = RAW_DIR / "drug_action_labels.csv"
    if action_path.exists():
        try:
            action_df = pd.read_csv(action_path, dtype=str)
            action_df["drug_name"] = action_df["drug_name"].fillna("").map(normalize_text)
            action_df["action_class"] = action_df["action_class"].fillna("other").map(normalize_text)
        except EmptyDataError:
            action_df = pd.DataFrame(columns=["drug_name", "action_class"])
    else:
        action_df = pd.DataFrame(columns=["drug_name", "action_class"])

    drugs = merged["drug_name"].drop_duplicates().to_frame()
    drugs = drugs.merge(action_df, on="drug_name", how="left")
    drugs["action_class"] = drugs["action_class"].where(
        drugs["action_class"].isin(ACTION_CLASSES), "other"
    )

    side_effect_index = pd.DataFrame(
        {"side_effect_name": sorted(top_side_effects), "side_effect_idx": range(len(top_side_effects))}
    )
    drug_index = pd.DataFrame(
        {"drug_name": sorted(drugs["drug_name"].unique()), "drug_idx": range(drugs["drug_name"].nunique())}
    )
    action_index = pd.DataFrame({"action_class": ACTION_CLASSES, "action_idx": range(len(ACTION_CLASSES))})

    edges = (
        merged.merge(drug_index, on="drug_name", how="inner")
        .merge(side_effect_index, on="side_effect_name", how="inner")
        .loc[:, ["drug_idx", "side_effect_idx"]]
        .drop_duplicates()
    )

    drug_targets = (
        drug_index.merge(drugs[["drug_name", "action_class"]], on="drug_name", how="left")
        .merge(action_index, on="action_class", how="left")
        .fillna({"action_idx": action_index.loc[action_index["action_class"] == "other", "action_idx"].iloc[0]})
        .loc[:, ["drug_idx", "action_idx"]]
    )
    drug_targets["action_idx"] = drug_targets["action_idx"].astype(int)

    drug_to_effects = edges.groupby("drug_idx")["side_effect_idx"].apply(list)
    all_drug_ids = drug_index["drug_idx"].to_numpy()
    has_min_labels = np.array([len(drug_to_effects.get(i, [])) >= 2 for i in all_drug_ids])
    eligible = all_drug_ids[has_min_labels]
    train_ids, test_ids = train_test_split(eligible, test_size=0.15, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)

    split_df = pd.DataFrame(
        {
            "drug_idx": np.concatenate([train_ids, val_ids, test_ids]),
            "split": ["train"] * len(train_ids) + ["val"] * len(val_ids) + ["test"] * len(test_ids),
        }
    )

    paths = {
        "drug_index": PROCESSED_DIR / "drug_index.csv",
        "side_effect_index": PROCESSED_DIR / "side_effect_index.csv",
        "action_index": PROCESSED_DIR / "action_index.csv",
        "drug_side_effect_edges": PROCESSED_DIR / "drug_side_effect_edges.csv",
        "drug_action_labels": PROCESSED_DIR / "drug_action_labels.csv",
        "splits": PROCESSED_DIR / "splits.csv",
    }
    drug_index.to_csv(paths["drug_index"], index=False)
    side_effect_index.to_csv(paths["side_effect_index"], index=False)
    action_index.to_csv(paths["action_index"], index=False)
    edges.to_csv(paths["drug_side_effect_edges"], index=False)
    drug_targets.to_csv(paths["drug_action_labels"], index=False)
    split_df.to_csv(paths["splits"], index=False)

    return paths
