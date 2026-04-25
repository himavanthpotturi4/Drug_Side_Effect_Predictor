from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

SIDER_URLS = {
    "meddra_all_se.tsv.gz": [
        "https://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz",
        "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz",
    ],
    "drug_names.tsv": [
        "https://sideeffects.embl.de/media/download/drug_names.tsv",
        "http://sideeffects.embl.de/media/download/drug_names.tsv",
    ],
}

ACTION_CLASSES = ["human_cell", "virus", "bacteria", "fungus", "other"]

# Keep dimensionality manageable while still informative.
TOP_SIDE_EFFECTS = 400
