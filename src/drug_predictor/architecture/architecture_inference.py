from __future__ import annotations

from difflib import get_close_matches

import pandas as pd
import torch
import requests
import re

from ..config import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from ..utils import load_json, normalize_text, save_json
from .module_3_knowledge_graph_construction import run_module_3_knowledge_graph_construction
from .module_4_graph_neural_network import ArchitectureGNN
from .module_6_link_prediction import run_module_6_link_prediction
from .module_7_drug_action_classification import run_module_7_drug_action_classification
from .module_9_output_visualization import run_module_9_output_visualization


class ArchitecturePredictor:
    def __init__(self):
        ckpt = torch.load(MODELS_DIR / "architecture_gnn.pt", map_location="cpu", weights_only=True)
        self.threshold = float(ckpt.get("threshold", 0.35))
        self._common_uses_cache_path = PROCESSED_DIR / "common_uses_cache.json"
        self._curated_common_uses = {
            "aspirin": ["Pain relief", "Fever reduction", "Blood clot prevention"],
            "ibuprofen": ["Pain relief", "Fever reduction", "Inflammation reduction"],
            "acetaminophen": ["Pain relief", "Fever reduction"],
            "paracetamol": ["Pain relief", "Fever reduction"],
            "metformin": ["Blood sugar control", "Type 2 diabetes management"],
            "amoxicillin": ["Bacterial infection treatment"],
            "atorvastatin": ["Cholesterol reduction", "Heart disease risk reduction"],
            "omeprazole": ["Acid reflux relief", "Ulcer treatment"],
            "losartan": ["Blood pressure control", "Kidney protection"],
            "amlodipine": ["Blood pressure control", "Chest pain management"],
        }

        self.drug_index = pd.read_csv(PROCESSED_DIR / "drug_index.csv")
        self.side_effect_index = pd.read_csv(PROCESSED_DIR / "side_effect_index.csv")
        self.action_index = pd.read_csv(PROCESSED_DIR / "action_index.csv")
        self.kg = run_module_3_knowledge_graph_construction(export_visualization=False)
        disease_path = RAW_DIR / "drug_disease_labels.csv"
        if disease_path.exists() and disease_path.stat().st_size > 0:
            disease_df = pd.read_csv(disease_path, dtype=str)
            disease_df["drug_name"] = disease_df["drug_name"].fillna("").map(normalize_text)
            disease_df["disease_name"] = disease_df["disease_name"].fillna("").map(normalize_text)
            disease_df = disease_df[(disease_df["drug_name"] != "") & (disease_df["disease_name"] != "")]
            self.drug_to_diseases = (
                disease_df.groupby("drug_name")["disease_name"].apply(lambda x: sorted(set(x))).to_dict()
            )
        else:
            self.drug_to_diseases = {}
        self._treated_for_cache: dict[str, list[str]] = {}
        if self._common_uses_cache_path.exists() and self._common_uses_cache_path.stat().st_size > 0:
            try:
                raw_cache = load_json(self._common_uses_cache_path)
                if isinstance(raw_cache, dict):
                    self._treated_for_cache = {
                        normalize_text(k): [str(item) for item in v[:3]]
                        for k, v in raw_cache.items()
                        if isinstance(v, list)
                    }
            except Exception:
                self._treated_for_cache = {}

        self.model = ArchitectureGNN(
            num_nodes=ckpt["num_nodes"],
            num_drugs=ckpt["num_drugs"],
            num_side_effects=ckpt["num_side_effects"],
            num_actions=ckpt["num_actions"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self._name_to_idx = dict(zip(self.drug_index["drug_name"], self.drug_index["drug_idx"]))
        with torch.no_grad():
            side_logits, action_logits, _ = self.model(self.kg.adjacency)
        self._side_effect_logits = side_logits.detach().cpu()
        self._action_logits = action_logits.detach().cpu()

    def _save_common_uses_cache(self) -> None:
        try:
            save_json(self._common_uses_cache_path, self._treated_for_cache)
        except Exception:
            pass

    def _chembl_get(self, endpoint: str, params: dict | None = None, timeout: int = 4) -> dict:
        base = "https://www.ebi.ac.uk/chembl/api/data"
        resp = requests.get(f"{base}/{endpoint}", params=params or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _fetch_treated_for_top3(self, drug_name: str) -> list[str]:
        atc_uses = self._fetch_chembl_atc_uses_top3(drug_name)
        if atc_uses:
            return atc_uses

        openfda = self._fetch_openfda_indications_top3(drug_name)
        if openfda:
            return openfda

        # Prefer approved indications (phase 4), then phase 3 if needed.
        try:
            payload = self._chembl_get("molecule/search.json", {"q": drug_name, "limit": 5})
            molecules = payload.get("molecules", [])
            if not molecules:
                return []

            norm = normalize_text(drug_name)
            chembl_id = None
            for m in molecules:
                if normalize_text(m.get("pref_name") or "") == norm:
                    chembl_id = m.get("molecule_chembl_id")
                    break
            if chembl_id is None:
                chembl_id = molecules[0].get("molecule_chembl_id")
            if not chembl_id:
                return []

            def collect(min_phase: int) -> list[tuple[str, int]]:
                p = self._chembl_get(
                    "drug_indication.json",
                    {
                        "molecule_chembl_id": chembl_id,
                        "limit": 200,
                        "max_phase_for_ind__gte": min_phase,
                    },
                )
                items = []
                for row in p.get("drug_indications", []):
                    name = normalize_text(row.get("mesh_heading") or row.get("efo_term") or "")
                    phase = int(row.get("max_phase_for_ind") or 0)
                    if name:
                        items.append((name, phase))
                return items

            indications = collect(4)
            if len(indications) < 3:
                indications.extend(collect(3))

            # Keep highest phase first, dedupe names, and remove obvious noisy meta terms.
            noise = {"aging", "disease", "symptom"}
            indications = sorted(indications, key=lambda x: x[1], reverse=True)
            out: list[str] = []
            seen = set()
            for name, _ in indications:
                if name in noise or name in seen:
                    continue
                if "," in name:
                    # Often niche/over-specific strings in this dataset view.
                    continue
                seen.add(name)
                out.append(name)
                if len(out) == 3:
                    break
            return out
        except Exception:
            return []

    def _fetch_chembl_atc_uses_top3(self, drug_name: str) -> list[str]:
        try:
            payload = self._chembl_get("molecule/search.json", {"q": drug_name, "limit": 5})
            molecules = payload.get("molecules", [])
            if not molecules:
                return []

            norm = normalize_text(drug_name)
            picked = None
            for m in molecules:
                if normalize_text(m.get("pref_name") or "") == norm:
                    picked = m
                    break
            if picked is None:
                picked = molecules[0]

            atc_codes = picked.get("atc_classifications", []) or []
            if not atc_codes:
                return []

            out = []
            seen = set()
            for code in atc_codes[:6]:
                try:
                    data = self._chembl_get(f"atc_class/{code}.json")
                except Exception:
                    continue

                # Prefer more human-readable mid-level therapeutic descriptions.
                candidates = [
                    data.get("level2_description"),
                    data.get("level3_description"),
                    data.get("level1_description"),
                ]
                for c in candidates:
                    c = normalize_text(c or "")
                    if not c or c in seen:
                        continue
                    seen.add(c)
                    out.append(c.capitalize())
                    break
                if len(out) == 3:
                    break
            return out
        except Exception:
            return []

    def _fetch_openfda_indications_top3(self, drug_name: str) -> list[str]:
        try:
            q = normalize_text(drug_name)
            resp = requests.get(
                "https://api.fda.gov/drug/label.json",
                params={
                    "search": f'openfda.generic_name:"{q}"+OR+openfda.brand_name:"{q}"',
                    "limit": 5,
                },
                timeout=4,
            )
            if resp.status_code != 200:
                return []
            payload = resp.json()
            results = payload.get("results", [])
            if not results:
                return []

            picked = None
            for r in results:
                names = []
                openfda = r.get("openfda", {})
                names.extend([normalize_text(x) for x in openfda.get("generic_name", [])])
                names.extend([normalize_text(x) for x in openfda.get("brand_name", [])])
                if q in names or any(q in n or n in q for n in names):
                    picked = r
                    break
            if picked is None:
                picked = results[0]

            text_blocks = picked.get("indications_and_usage", [])
            if not text_blocks:
                return []

            raw = " ".join(text_blocks).replace("\n", " ").strip()
            raw = re.sub(r"\s+", " ", raw)
            raw = re.sub(r"\b(indications?\s*&?\s*usage)\b[:\s]*", " ", raw, flags=re.I)
            raw = re.sub(r"\b\d+\s*", " ", raw)

            # Try to extract concise phrases after "indicated for" / "used for" patterns.
            phrases = []
            patterns = [
                r"indicated for ([^.;]{8,120})",
                r"used for ([^.;]{8,120})",
                r"for the treatment of ([^.;]{8,120})",
                r"for relief of ([^.;]{8,120})",
            ]
            low = raw.lower()
            for pat in patterns:
                for m in re.finditer(pat, low, flags=re.I):
                    txt = m.group(1).strip(" ,;-")
                    txt = re.sub(r"\bpatients?\s+with\b", "", txt).strip(" ,;-")
                    txt = re.sub(r"\b(adults?|children|pediatric patients?)\b", "", txt, flags=re.I).strip(" ,;-")
                    if 4 <= len(txt) <= 80:
                        phrases.append(txt)

            if not phrases:
                # Fallback: take first clean sentence fragments.
                for part in re.split(r"[.;]", raw):
                    p = part.strip(" ,;-")
                    p = re.sub(r"^(this product|drug|medicine)\s+(is|are)\s+", "", p, flags=re.I)
                    if 8 <= len(p) <= 80:
                        phrases.append(p.lower())

            noise = {"safety coated", "ask your doctor", "temporary relief"}
            out = []
            seen = set()
            for p in phrases:
                p = re.sub(r"\s+", " ", p).strip()
                if not p:
                    continue
                if any(n in p for n in noise):
                    continue
                if p in seen:
                    continue
                seen.add(p)
                out.append(p.capitalize())
                if len(out) == 3:
                    break
            return out
        except Exception:
            return []

    def _resolve(self, name: str) -> tuple[str | None, int | None]:
        key = normalize_text(name)
        if key in self._name_to_idx:
            return key, int(self._name_to_idx[key])
        match = get_close_matches(key, self._name_to_idx.keys(), n=1, cutoff=0.82)
        if not match:
            return None, None
        best = match[0]
        return best, int(self._name_to_idx[best])

    def _get_local_treated_for(self, drug_name: str) -> list[str]:
        local = self.drug_to_diseases.get(drug_name, [])
        if not local:
            return []
        cleaned = [item for item in local if item and item != "unknown_disease"]
        return cleaned[:3]

    def _get_treated_for(self, drug_name: str) -> list[str]:
        cached = self._treated_for_cache.get(drug_name)
        if cached:
            return cached

        local = self._get_local_treated_for(drug_name)
        if local:
            self._treated_for_cache[drug_name] = local
            self._save_common_uses_cache()
            return local

        curated = self._curated_common_uses.get(drug_name, [])
        self._treated_for_cache[drug_name] = curated
        self._save_common_uses_cache()
        return curated

    def predict(self, drug_name: str, top_k: int = 15) -> dict:
        resolved, drug_idx = self._resolve(drug_name)
        if resolved is None:
            return {"found": False, "message": "Drug not found in trained vocabulary."}

        predicted_side_effects = run_module_6_link_prediction(
            side_effect_logits=self._side_effect_logits,
            side_effect_index=self.side_effect_index,
            drug_idx=drug_idx,
            top_k=top_k,
            threshold=self.threshold,
        )
        predicted_action = run_module_7_drug_action_classification(
            action_logits=self._action_logits,
            action_index=self.action_index,
            drug_idx=drug_idx,
        )
        payload = run_module_9_output_visualization(
            input_drug=drug_name,
            resolved_drug=resolved,
            predicted_action=predicted_action,
            predicted_side_effects=predicted_side_effects,
        )
        payload["treated_for"] = self._get_treated_for(resolved)
        return {"found": True, **payload}
