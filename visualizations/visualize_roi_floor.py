from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from data_loading.roi_floor import RoIFloorConfig, create_roi_floor
from data_loading.sensfloor_dataset import DatasetConfig, load_single_dataset

DATA_PATH = Path("./data/2025-12-01_12-44-43")
VIDEO_PATH = DATA_PATH / "video.mp4"


floor_config = RoIFloorConfig(
    x_size=6,
    y_size=4,
    history_maxlen=10,
    roi_size=3,
    active_field_min_value=135,
)

dataset_config = DatasetConfig(
    floor_config=floor_config,
    normalize_signals=False,
)

# Define figure parameters
num_rows = int(np.ceil(floor_config.history_maxlen / 5))
num_cols = 5
figsize = (20, 8)

# Load dataset
dataset = load_single_dataset(data_path=DATA_PATH, config=dataset_config)
frame_number = dataset.frames_containing_messages[0]

# Get images from video
cap = cv2.VideoCapture(str(VIDEO_PATH))
fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
for i, ax in enumerate(axs.flatten()):
    current_frame_number = (frame_number - floor_config.history_maxlen + 1) + i
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ax.imshow(frame_rgb)
    ax.axis("off")

fig.tight_layout()


# Get floor signals
floor = create_roi_floor(config=floor_config, sensfloor_readout=dataset.sensfloor_readout_df, frame_number=frame_number)
roi = floor.get_roi()

if roi is None:
    message = "No RoI found. Does the floor contain activated fields?"
    raise RuntimeError(message)

floor_history = floor.history
x_size, y_size = floor_history.shape[1], floor_history.shape[2]

cmap = mcolors.LinearSegmentedColormap.from_list("signal colormap", ["#FFFFFF", "#0033FF", "#FF6A00"])

# Create plots
fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
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

    linestyle = "-" if timestamp == (floor_config.history_maxlen - 1) else "--"
    linewidth = 1.5 if timestamp == (floor_config.history_maxlen - 1) else 1
    roi_rect = patches.Rectangle(
        (roi_col_start, roi_row_start),
        floor_config.roi_size * 4,
        floor_config.roi_size * 4,
        linewidth=linewidth,
        edgecolor="red",
        facecolor="none",
        linestyle=linestyle,
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
