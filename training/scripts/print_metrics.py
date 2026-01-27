import argparse
from pathlib import Path

import pandas as pd

from training.configs import (
    CONFIG_FILE_NAME,
    TrainingConfiguration,
)
from training.trainer.base_trainer import METRICS_FILE_NAME
from training.trainer.sensfloor_trainer import SensfloorTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model folder", type=Path)
    return parser.parse_args()


def main(model_dir: Path) -> None:
    metrics_csv = model_dir / METRICS_FILE_NAME
    configuration = TrainingConfiguration.load(model_dir / CONFIG_FILE_NAME)
    df = pd.read_csv(metrics_csv)
    for row in df.to_dict(orient="records"):
        SensfloorTrainer.log_training_metrics(row, configuration.landmarks)


if __name__ == "__main__":
    args = parse_args()
    main(args.model)
