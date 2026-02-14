import argparse
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage

from src.data_loading.roi_floor import RoIFloorConfig
from src.definitions import VIDEO_FILENAME
from src.training.configs import DatasetType
from src.training.dataset.dataset_utils import DatasetConfig
from src.training.dataset.load_data import load_single_dataset


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


def draw_roi(ax: Axes, x: int, y: int, roi_size: int) -> patches.Patch:
    roi_row_start = x * 4
    roi_col_start = y * 4

    roi_rect = patches.Rectangle(
        (roi_col_start, roi_row_start),
        roi_size * 4,
        roi_size * 4,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
        linestyle="-",
        zorder=10,
    )
    return ax.add_patch(roi_rect)


def main(data_dir: Path, data_index: int) -> None:
    print(f"Visualize data from directory: {data_dir}")
    video_path = data_dir / VIDEO_FILENAME
    floor_config = RoIFloorConfig(
        x_size=6,
        y_size=4,
        history_maxlen=15,
        roi_size=4,
        active_field_min_value=140,
        remove_noise=True,
    )

    dataset_config = DatasetConfig(
        floor_config=floor_config,
        normalize_signals=False,
    )

    # Load dataset
    dataset = load_single_dataset(data_path=data_dir, config=dataset_config, dataset_type=DatasetType.HISTORY)
    data = dataset.get_detailed_data(data_index)
    roi = data.floor.get_roi()

    if roi is None:
        message = "No RoI found. Does the floor contain activated fields?"
        raise RuntimeError(message)

    # Define figure parameters
    num_rows = int(np.ceil(floor_config.history_maxlen / 5))
    num_cols = 5
    figsize = (20, 8)

    # Get images from video
    cap = cv2.VideoCapture(str(video_path))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    for timestamp, ax in enumerate(axs.flatten()):
        current_frame_number = (data.frame_number - len(axs.flatten()) + 1) + timestamp
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ax.set_title(f"Frame {current_frame_number}")
        ax.imshow(frame_rgb)
        ax.axis("off")

    fig.tight_layout()

    # Create plots
    cmap = mcolors.LinearSegmentedColormap.from_list("signal colormap", ["#FFFFFF", "#000000"])
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    image = None
    for timestamp, ax in enumerate(axs.flatten()):
        image = draw_floor(ax, data.floor.history[timestamp], cmap)

        if floor_config.roi_size is not None:
            _ = draw_roi(ax, roi.x, roi.y, floor_config.roi_size)

        ax.set_title(f"ROI at frame {(data.frame_number - len(axs.flatten()) + 1) + timestamp}")
    if image:
        plt.tight_layout(rect=(0, 0, 0.9, 1))
        cbar_ax = fig.add_axes((0.91, 0.15, 0.02, 0.7))
        fig.colorbar(image, cax=cbar_ax)
    else:
        plt.tight_layout()

    plt.show()


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


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.data_index)
