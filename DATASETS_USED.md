# Datasets Used In This Project

## 1) SIDER 4.1 (Drug Side Effects)
- Source: `https://sideeffects.embl.de/download/`
- Files used in workspace:
  - `data/raw/meddra_all_se.tsv.gz`
  - `data/raw/drug_names.tsv`
- Purpose:
  - Build drug-side effect relationships.
  - Build normalized drug vocabulary.

## 2) ChEMBL Web Services (Drug Targets + Indications)
- Source docs: `https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services`
- API base used by code:
  - `https://www.ebi.ac.uk/chembl/api/data`
- Endpoints used:
  - molecule search
  - mechanism
  - target
  - drug_indication
  - atc_class
- Purpose:
  - Drug action class labeling (`human_cell`, `virus`, `bacteria`, `fungus`, `other` internally).
  - Drug-disease indication enrichment.
  - Clean therapeutic "Common Uses" via ATC categories.

## 3) Derived Datasets Generated In This Repo
- `data/raw/drug_action_labels.csv`
- `data/raw/drug_disease_labels.csv`
- `data/raw/drug_symptom_labels.csv`
- `data/processed/drug_index.csv`
- `data/processed/side_effect_index.csv`
- `data/processed/symptom_index.csv`
- `data/processed/disease_index.csv`
- `data/processed/drug_side_effect_edges.csv`
- `data/processed/drug_symptom_edges.csv`
- `data/processed/drug_disease_edges.csv`
- `data/processed/drug_action_labels.csv`

## Current Coverage In Your Workspace
- Unique drugs in source list: ~1344
- Trained drug vocabulary: ~1340
- Action labels collected: 1344 rows
