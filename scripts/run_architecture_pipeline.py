import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from drug_predictor.architecture.module_1_data_collection import run_module_1_data_collection
from drug_predictor.architecture.module_2_data_preprocessing import run_module_2_data_preprocessing
from drug_predictor.architecture.module_5_model_training import run_module_5_model_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-drugs", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print("Module 1: Data Collection")
    run_module_1_data_collection(max_drugs=args.max_drugs)
    print("Module 2: Data Preprocessing")
    run_module_2_data_preprocessing()
    print("Module 3+4+5: KG + GNN + Training")
    metrics = run_module_5_model_training(epochs=args.epochs)
    print("Completed architecture pipeline.")
    print(metrics)


if __name__ == "__main__":
    main()

