from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

from ..config import RAW_DIR
from ..data_collection import build_drug_action_lookup, download_sider
from ..preprocessing import load_sider_tables
from ..utils import normalize_text


def _chembl_get(endpoint: str, params: dict | None = None) -> dict:
    base = "https://www.ebi.ac.uk/chembl/api/data"
    response = requests.get(f"{base}/{endpoint}", params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def _find_best_molecule_chembl_id(drug_name: str) -> str | None:
    payload = _chembl_get("molecule/search.json", {"q": drug_name, "limit": 5})
    molecules = payload.get("molecules", [])
    if not molecules:
        return None
    normalized = normalize_text(drug_name)
    for molecule in molecules:
        pref = normalize_text(molecule.get("pref_name") or "")
        if pref == normalized:
            return molecule.get("molecule_chembl_id")
    return molecules[0].get("molecule_chembl_id")


def _load_drug_names() -> list[str]:
    raw = pd.read_csv(RAW_DIR / "drug_names.tsv", sep="\t", header=None, dtype=str)
    if raw.shape[1] >= 4:
        raw = raw.iloc[:, :4]
        raw.columns = ["stitch_flat", "stitch_stereo", "pubchem_id", "drug_name"]
    elif raw.shape[1] == 2:
        raw.columns = ["stitch_flat", "drug_name"]
    else:
        raise ValueError("Unsupported drug_names.tsv format.")
    names = raw["drug_name"].fillna("").map(normalize_text)
    return names[names != ""].drop_duplicates().tolist()


def build_drug_disease_lookup(max_drugs: int | None = None, delay_seconds: float = 0.05) -> pd.DataFrame:
    out_path = RAW_DIR / "drug_disease_labels.csv"
    drug_names = _load_drug_names()
    if max_drugs is not None:
        drug_names = drug_names[:max_drugs]

    records = []
    for drug in tqdm(drug_names, desc="Fetching ChEMBL indications"):
        disease_names: set[str] = set()
        try:
            chembl_id = _find_best_molecule_chembl_id(drug)
            if chembl_id:
                payload = _chembl_get(
                    "drug_indication.json",
                    {"molecule_chembl_id": chembl_id, "limit": 200},
                )
                for row in payload.get("drug_indications", []):
                    disease = normalize_text(row.get("mesh_heading", ""))
                    if disease:
                        disease_names.add(disease)
        except Exception:  # noqa: BLE001
            disease_names = set()

        if not disease_names:
            records.append({"drug_name": drug, "disease_name": "unknown_disease"})
        else:
            for disease in sorted(disease_names):
                records.append({"drug_name": drug, "disease_name": disease})
        time.sleep(delay_seconds)

    df = pd.DataFrame(records).drop_duplicates()
    df.to_csv(out_path, index=False)
    return df


def build_drug_symptom_lookup() -> pd.DataFrame:
    # Report-aligned dataset 3: drug-symptom data.
    se_df, names_df = load_sider_tables()
    se_df["side_effect_name"] = se_df["side_effect_name"].fillna("").map(normalize_text)
    names_df["drug_name"] = names_df["drug_name"].fillna("").map(normalize_text)
    merged = se_df.merge(names_df[["stitch_flat", "drug_name"]], on="stitch_flat", how="left")
    merged = merged[["drug_name", "side_effect_name"]].dropna().drop_duplicates()
    merged = merged[(merged["drug_name"] != "") & (merged["side_effect_name"] != "")]

    symptom_df = merged.rename(columns={"side_effect_name": "symptom_name"})
    out_path = RAW_DIR / "drug_symptom_labels.csv"
    symptom_df.to_csv(out_path, index=False)
    return symptom_df


def run_module_1_data_collection(max_drugs: int = 1200) -> dict[str, str]:
    download_sider()
    build_drug_action_lookup(max_drugs=max_drugs)
    build_drug_disease_lookup(max_drugs=max_drugs)
    build_drug_symptom_lookup()
    return {
        "drug_side_effect_source": str(RAW_DIR / "meddra_all_se.tsv.gz"),
        "drug_name_source": str(RAW_DIR / "drug_names.tsv"),
        "drug_action_source": str(RAW_DIR / "drug_action_labels.csv"),
        "drug_disease_source": str(RAW_DIR / "drug_disease_labels.csv"),
        "drug_symptom_source": str(RAW_DIR / "drug_symptom_labels.csv"),
    }
