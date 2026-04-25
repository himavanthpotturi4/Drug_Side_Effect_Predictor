from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

from .config import RAW_DIR, SIDER_URLS
from .utils import ensure_dir, normalize_text


def _download_with_fallback(urls: Iterable[str], output_path: Path, timeout: int = 60) -> None:
    last_error = None
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with output_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Failed downloading {output_path.name}: {last_error}")


def download_sider() -> None:
    ensure_dir(RAW_DIR)
    for filename, urls in SIDER_URLS.items():
        out = RAW_DIR / filename
        if out.exists():
            continue
        _download_with_fallback(urls, out)


def _classify_organism(organism: str) -> str:
    org = normalize_text(organism)
    if any(k in org for k in ["homo sapiens", "human"]):
        return "human_cell"
    if any(k in org for k in ["virus", "viridae", "influenza", "hiv", "sars", "hepatitis"]):
        return "virus"
    if any(k in org for k in ["bacter", "bacillus", "staphyl", "strept", "escherichia", "mycobacter"]):
        return "bacteria"
    if any(k in org for k in ["fung", "candida", "aspergillus", "yeast"]):
        return "fungus"
    return "other"


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


def _paginate_mechanisms(molecule_chembl_id: str) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while True:
        payload = _chembl_get(
            "mechanism.json",
            {"molecule_chembl_id": molecule_chembl_id, "limit": 100, "offset": offset},
        )
        chunk = payload.get("mechanisms", [])
        out.extend(chunk)
        page_meta = payload.get("page_meta", {})
        if not page_meta.get("next"):
            break
        offset += 100
    return out


def build_drug_action_lookup(
    max_drugs: int | None = None,
    delay_seconds: float = 0.05,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    drug_names_path = RAW_DIR / "drug_names.tsv"
    if not drug_names_path.exists():
        raise FileNotFoundError("Run download_sider() first.")

    raw = pd.read_csv(drug_names_path, sep="\t", header=None, dtype=str)
    if raw.shape[1] >= 4:
        raw = raw.iloc[:, :4]
        raw.columns = ["stitch_flat", "stitch_stereo", "pubchem_id", "drug_name"]
        drug_names = raw
    elif raw.shape[1] == 2:
        raw.columns = ["stitch_flat", "drug_name"]
        drug_names = raw
    else:
        raise ValueError("Unsupported drug_names.tsv format.")
    drug_names["drug_name"] = drug_names["drug_name"].fillna("").map(normalize_text)
    unique_drugs = (
        drug_names.loc[drug_names["drug_name"] != "", "drug_name"].drop_duplicates().tolist()
    )
    if max_drugs is not None:
        unique_drugs = unique_drugs[:max_drugs]

    out_path = RAW_DIR / "drug_action_labels.csv"
    existing = {}
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            old = pd.read_csv(out_path)
            for _, row in old.iterrows():
                existing[str(row["drug_name"])] = str(row["action_class"])
        except Exception:  # noqa: BLE001
            existing = {}

    target_cache: dict[str, str] = {}
    records = [{"drug_name": k, "action_class": v} for k, v in existing.items()]
    processed = set(existing.keys())
    pending = [d for d in unique_drugs if d not in processed]

    def checkpoint() -> None:
        pd.DataFrame(records).drop_duplicates(subset=["drug_name"]).to_csv(out_path, index=False)
    for idx, drug in enumerate(tqdm(pending, desc="Fetching ChEMBL actions"), start=1):
        try:
            chembl_id = _find_best_molecule_chembl_id(drug)
            if not chembl_id:
                records.append({"drug_name": drug, "action_class": "other"})
                continue

            mechs = _paginate_mechanisms(chembl_id)
            organisms = []
            for m in mechs:
                target_id = m.get("target_chembl_id")
                if not target_id:
                    continue
                if target_id in target_cache:
                    org = target_cache[target_id]
                else:
                    t_payload = _chembl_get(f"target/{target_id}.json")
                    org = t_payload.get("organism", "")
                    target_cache[target_id] = org
                if org:
                    organisms.append(org)
                time.sleep(delay_seconds)

            if not organisms:
                action = "other"
            else:
                labels = [_classify_organism(org) for org in organisms]
                action = pd.Series(labels).mode().iloc[0]

            records.append({"drug_name": drug, "action_class": action})
        except Exception:  # noqa: BLE001
            records.append({"drug_name": drug, "action_class": "other"})
        time.sleep(delay_seconds)
        if idx % checkpoint_every == 0:
            checkpoint()

    checkpoint()
    df = pd.DataFrame(records).drop_duplicates(subset=["drug_name"])
    return df


def run_data_collection(max_drugs: int | None = 3500) -> None:
    download_sider()
    build_drug_action_lookup(max_drugs=max_drugs)
