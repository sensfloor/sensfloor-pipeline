import argparse
from pathlib import Path

from training.configs import TrainingConfiguration
from training.training_pipeline import CONFIG_FILE_NAME, run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model folder", type=Path)
    return parser.parse_args()


def main(model_dir: Path) -> None:
    configuration = TrainingConfiguration.load(model_dir / CONFIG_FILE_NAME)
    run_config(do_train=False, do_test=True, configuration=configuration)


if __name__ == "__main__":
    args = parse_args()

    main(args.model)
