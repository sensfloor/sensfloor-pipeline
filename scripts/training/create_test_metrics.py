import argparse
from pathlib import Path

import torch

from src.definitions import TEST_METRICS_FILENAME, BEST_MODEL_FILENAME
from src.training.configs import (CONFIG_FILE_NAME, TrainingConfiguration)
from src.training.trainer.sensfloor_trainer import (get_test_metrics)
from src.training.training_pipeline import get_training_setup


def get_and_save_test_metrics(configuration: TrainingConfiguration, model_folder: Path) -> None:
    model, device, trainer, train_loader, val_loader, test_loader = get_training_setup(configuration, test_batch_size=1)

    model_file_name = BEST_MODEL_FILENAME
    model_path = model_folder / model_file_name

    checkpoint = torch.load(f=model_path)
    model.load_state_dict(state_dict=checkpoint)

    test_metrics = get_test_metrics(model, test_loader, device, trainer)
    trainer.save_metrics_to_csv(test_metrics[-1].keys(), test_metrics, file_name=TEST_METRICS_FILENAME)

def main(model_path: Path):
    config = TrainingConfiguration.load(model_path / CONFIG_FILE_NAME)
    get_and_save_test_metrics(config, model_path)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live prediction of poses")

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model folder containing weights and configuration file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.model)