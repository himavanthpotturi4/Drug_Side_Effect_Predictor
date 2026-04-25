import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from drug_predictor.data_collection import run_data_collection
from drug_predictor.preprocessing import build_training_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-drugs", type=int, default=1200)
    args = parser.parse_args()

    run_data_collection(max_drugs=args.max_drugs)
    build_training_tables()
    print("Data collection and preprocessing finished.")


if __name__ == "__main__":
    main()
