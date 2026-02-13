import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_loading.pose_landmark import LINKS, PoseLandmark
from src.inference.model_loading import load_floor, load_model
from src.training.configs import CONFIG_FILE_NAME, DatasetType, TrainingConfiguration
from src.training.dataset.load_data import load_single_dataset
from src.training.dataset.sensfloor_dataset import DatasetConfig


def map_ground_truth_to_model_joints(pose: np.ndarray, landmarks: list[PoseLandmark]) -> np.ndarray:
    return pose[[landmark.value for landmark in landmarks]].copy()


def get_links_to_plot(landmarks: list[PoseLandmark]) -> list[tuple[int, int]]:
    return [(landmarks.index(l1), landmarks.index(l2)) for (l1, l2) in LINKS if (l1 in landmarks and l2 in landmarks)]


def add_pose_to_ax(ax: Any, pose: np.ndarray, color: str, links: list[tuple[int, int]]) -> None:  # noqa: ANN401
    pose_to_plot = pose.copy()
    pose_to_plot[:, 1] *= -1
    ax.scatter(pose_to_plot[:, 2], pose_to_plot[:, 0], pose_to_plot[:, 1], c=color, marker="o")

    # Draw bones
    for start, end in links:
        ax.plot(
            [pose_to_plot[start, 2], pose_to_plot[end, 2]],
            [pose_to_plot[start, 0], pose_to_plot[end, 0]],
            [pose_to_plot[start, 1], pose_to_plot[end, 1]],
            c=color,
        )


def create_pose_comparison(
    ground_truth_pose: np.ndarray,
    predicted_pose: np.ndarray,
    links: list[tuple[int, int]],
) -> None:
    save_path = Path("outputs/images/pose_comparison")
    save_path.mkdir(exist_ok=True, parents=True)
    rotations = [0, 90, 180, 270]
    for rotation in rotations:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        add_pose_to_ax(ax, ground_truth_pose, color="black", links=links)
        add_pose_to_ax(ax, predicted_pose, color="#ff0000", links=links)
        setup_clean_ax(ax)
        ax.view_init(elev=5, azim=rotation)

        figure_save_path = save_path / f"{rotation}_degree_rotation.svg"
        plt.savefig(figure_save_path, format="svg", bbox_inches="tight")
        print(f"Saved plot to {figure_save_path}")


def setup_clean_ax(ax: Any) -> None:  # noqa: ANN401
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_aspect("equal")
    ax.set_axis_off()


@torch.no_grad()
def main(model_folder: Path, data_folder: Path, data_index: int) -> None:
    config = TrainingConfiguration.load(model_folder / CONFIG_FILE_NAME)
    model, device, _ = load_model(model_folder)
    _, floor_config = load_floor(model_folder)

    dataset_config = DatasetConfig(
        floor_config=floor_config,
        rotate_data=config.rotate_data,
        normalize_signals=config.do_normalize,
        normalize_to_max=config.normalize_to_max,
    )

    dataset = load_single_dataset(
        data_path=data_folder,
        config=dataset_config,
        dataset_type=DatasetType.HISTORY,
    )

    data = dataset.get_detailed_data(data_index)

    x = data.transformed_roi_tensor.unsqueeze(0).to(device)
    predicted_pose = model(x)[0].reshape(-1, 3).cpu().numpy()

    # Rotate mediapipe skeleton landmarks
    pose = data.transformed_label_tensor.reshape(33, 3).numpy()
    ground_truth_pose = map_ground_truth_to_model_joints(pose, config.landmarks)
    links = get_links_to_plot(config.landmarks)
    create_pose_comparison(ground_truth_pose, predicted_pose, links)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize difference between estimated and ground truth pose")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to sequence data folder.",
    )
    parser.add_argument(
        "--data-index",
        type=int,
        required=True,
        help="Index of data to use for estimating the pose",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model folder containing weights and configuration file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.model, args.data, args.data_index)
