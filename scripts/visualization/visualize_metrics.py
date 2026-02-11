import argparse
from pathlib import Path

from src.definitions import TEST_METRICS_FILENAME
from pathlib import Path

from src.visualization.visualize_metrics import load_and_plot_mjpe_professional


def main(model_path: Path):
    load_and_plot_mjpe_professional(model_path / TEST_METRICS_FILENAME)

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