import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage

from src.data_loading.roi_floor import RoIFloorConfig
from src.training.configs import DatasetType
from src.training.dataset.load_data import load_single_dataset
from src.training.dataset.sensfloor_dataset import DatasetConfig

FRAME_TO_SHOW = 906
CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]


def draw_floor(ax: Axes, floor: np.ndarray, cmap: Colormap, vmin: int = 127, vmax: int = 255) -> AxesImage:
    x_size_floor_array, y_size_floor_array = floor.shape

    image = ax.imshow(
        floor,
        origin="lower",
        cmap=cmap,
        extent=(0, y_size_floor_array, 0, x_size_floor_array),
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    ax.grid(which="major", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    fields_per_patch = 4
    x_positions = np.arange(0, y_size_floor_array + 1, fields_per_patch)
    y_positions = np.arange(0, x_size_floor_array + 1, fields_per_patch)

    x_labels = [str(int(x_position / fields_per_patch)) for x_position in x_positions]
    y_labels = [str(int(y_position / fields_per_patch)) for y_position in y_positions]

    ax.set_xticks(x_positions, x_labels)
    ax.set_yticks(y_positions, y_labels)

    ax.set_xlabel("y")
    ax.set_ylabel("x")

    ax.invert_yaxis()

    return image


def visualize_rotation(data_dir: Path, data_index: int) -> None:
    floor_config = RoIFloorConfig(x_size=6, y_size=4, history_maxlen=10, roi_size=3)

    dataset_config = DatasetConfig(
        floor_config=floor_config,
        normalize_signals=False,
        rotate_data=True,
    )

    dataset = load_single_dataset(
        data_path=data_dir,
        config=dataset_config,
        dataset_type=DatasetType.HISTORY,
    )

    data = dataset.get_detailed_data(data_index)

    # Rotate mediapipe skeleton landmarks
    rotated_t = data.transformed_label_tensor.reshape(33, 3).T

    plot_xs = rotated_t[0, :].tolist()  # mediapipe X is horizontal axis
    plot_ys = rotated_t[2, :].tolist()  # mediapipe Z is vertical axis
    plot_zs = [-y for y in rotated_t[1, :].tolist()]  # mediapipe Y is depth (and inverted)

    floor_tensor = data.transformed_roi_tensor[-1].numpy()

    fig, axes = plt.subplot_mosaic(
        [["Pose", "Floor"]],
        per_subplot_kw={
            "Pose": {"projection": "3d"},
            "Floor": {},
        },
        figsize=(15, 7),
    )

    # Plot landmarks
    ax_pose = axes["Pose"]
    ax_pose.scatter(plot_xs, plot_ys, plot_zs, c="r", marker="o")

    # Draw bones
    for start, end in CONNECTIONS:
        ax_pose.plot(
            [plot_xs[start], plot_xs[end]],
            [plot_ys[start], plot_ys[end]],
            [plot_zs[start], plot_zs[end]],
            c="blue",
        )

    ax_pose.set_xlabel("X")
    ax_pose.set_ylabel("Y (Z of mediapipe)")
    ax_pose.set_zlabel("Z (-Y of mediapipe)")
    ax_pose.set_title("Pose")

    # Set axis limits
    all_coords = plot_xs + plot_ys + plot_zs
    max_range = (max(all_coords) - min(all_coords)) / 2.0
    mid_x = (max(plot_xs) + min(plot_xs)) * 0.5
    mid_y = (max(plot_ys) + min(plot_ys)) * 0.5
    mid_z = (max(plot_zs) + min(plot_zs)) * 0.5
    ax_pose.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_pose.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_pose.set_zlim(mid_z - max_range, mid_z + max_range)

    # Adjust point of view
    ax_pose.view_init(elev=5, azim=110)

    ax_floor = axes["Floor"]
    cmap = mcolors.LinearSegmentedColormap.from_list("signal colormap", ["#FFFFFF", "#0033FF", "#FF6A00"])
    draw_floor(ax_floor, floor_tensor, cmap)

    fig.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize collected video and readout")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Required path to data directory containing video and sensfloor readout.",
    )
    parser.add_argument(
        "--data-index",
        type=int,
        required=True,
        help="Index of data to visualize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_rotation(args.data, args.data_index)
    visualize_rotation(args.data, args.data_index)
    visualize_rotation(args.data, args.data_index)
    visualize_rotation(args.data, args.data_index)
    plt.show()


if __name__ == "__main__":
    main()
