from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from data_loading.load_data import load_data
from data_loading.roi_floor import RoIFloorConfig, create_roi_floor
from data_loading.sensfloor_dataset import DatasetConfig

DATA_PATH = Path("./data/2025-12-02_12-30-55")
SENSFLOOR_READOUT_PATH = DATA_PATH / "sensfloor_readout.csv"
VIDEO_POSES_PATH = DATA_PATH / "video_poses.csv"


floor_config = RoIFloorConfig(
    x_size=6,
    y_size=4,
    history_maxlen=10,
    roi_size=3,
)

dataset_config = DatasetConfig(
    floor_config=floor_config,
    normalize_signals=False,
)

full_dataset = load_data(dataset_config)

print(f"Cumulative dataset contains {len(full_dataset.datasets)} datasets")
print(f"Dataset sizes: {full_dataset.cumulative_sizes}")

# Get first dataset that has at least 100 unique frames containing messages with poses
min_number_frames_with_poses = 100
dataset = next(
    (ds for ds in full_dataset.datasets if len(ds.frames_containing_messages) > min_number_frames_with_poses),
    None,
)

if dataset is None:
    message = "No dataset could be loaded... Do you have a data folder with all the required files?"
    raise RuntimeError(message)

frame_number = dataset.frames_containing_messages[200]
floor = create_roi_floor(config=floor_config, sensfloor_readout=dataset.sensfloor_readout_df, frame_number=frame_number)
roi = floor.get_roi()

if roi is None:
    message = "No RoI found. Does the floor contain activated fields?"
    raise RuntimeError(message)

floor_history = floor.history
x_size, y_size = floor_history.shape[1], floor_history.shape[2]

cmap = mcolors.LinearSegmentedColormap.from_list("signal colormap", ["#FFFFFF", "#0033FF", "#FF6A00"])

# Create plots
rows = int(np.ceil(floor_history.shape[0] / 5))
fig, axs = plt.subplots(rows, 5, figsize=(20, 8))
image = None
for timestamp, ax in enumerate(axs.flatten()):
    image = ax.imshow(
        floor_history[timestamp],
        origin="lower",
        cmap=cmap,
        extent=(0, y_size, 0, x_size),
        aspect="equal",
        vmin=127,
        vmax=255,
    )

    ax.grid(which="major", color="w", linestyle="-", linewidth=0.5, alpha=0.3)

    # Draw region of interest
    roi_row_start, roi_col_start = roi.x * 4, roi.y * 4

    roi_rect = patches.Rectangle(
        (roi_col_start, roi_row_start),
        floor_config.roi_size * 4,
        floor_config.roi_size * 4,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(roi_rect)

    # Adjust axis labels
    fields_per_patch = 4
    x_positions = np.arange(0, y_size + 1, fields_per_patch)
    y_positions = np.arange(0, x_size + 1, fields_per_patch)

    x_labels = [str(int(x_position / fields_per_patch)) for x_position in x_positions]
    y_labels = [str(int(y_position / fields_per_patch)) for y_position in y_positions]

    ax.set_xticks(x_positions, x_labels)
    ax.set_yticks(y_positions, y_labels)

    ax.set_title(f"ROI at {timestamp - (floor_config.history_maxlen - 1)}")
    ax.set_xlabel("y")
    ax.set_ylabel("x")

if image:
    plt.tight_layout(rect=(0, 0, 0.9, 1))

    cbar_ax = fig.add_axes((0.91, 0.15, 0.02, 0.7))
    fig.colorbar(image, cax=cbar_ax)
else:
    plt.tight_layout()

plt.show()
