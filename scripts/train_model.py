import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from drug_predictor.train import train_and_save


def main() -> None:
    metrics = train_and_save()
    print("Training completed.")
    print(metrics)


if __name__ == "__main__":
    main()
