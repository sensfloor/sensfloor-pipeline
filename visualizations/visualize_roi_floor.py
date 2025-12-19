from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from data_loading.roi_floor import RoIFloorConfig
from data_loading.sensfloor_dataset import DatasetConfig, load_single_dataset
from visualizations.utils import draw_floor, draw_roi

DATA_PATH = Path("./data/2025-12-02_12-24-03")
VIDEO_PATH = DATA_PATH / "video.mp4"

floor_config = RoIFloorConfig(
    x_size=6,
    y_size=4,
    history_maxlen=10,
    roi_size=3,
    active_field_min_value=140,
)

dataset_config = DatasetConfig(
    floor_config=floor_config,
    normalize_signals=False,
)

# Load dataset
dataset = load_single_dataset(data_path=DATA_PATH, config=dataset_config)
data = dataset.get_detailed_data(10)
roi = data.floor.get_roi()

if roi is None:
    message = "No RoI found. Does the floor contain activated fields?"
    raise RuntimeError(message)


# Define figure parameters
num_rows = int(np.ceil(floor_config.history_maxlen / 5))
num_cols = 5
figsize = (20, 8)

# Get images from video
cap = cv2.VideoCapture(str(VIDEO_PATH))
fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
for timestamp, ax in enumerate(axs.flatten()):
    current_frame_number = (data.frame_number - len(axs.flatten()) + 1) + timestamp
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ax.set_title(f"Frame {current_frame_number}")
    ax.imshow(frame_rgb)
    ax.axis("off")

fig.tight_layout()

# Create plots
cmap = mcolors.LinearSegmentedColormap.from_list("signal colormap", ["#FFFFFF", "#0033FF", "#FF6A00"])
fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
image = None
for timestamp, ax in enumerate(axs.flatten()):
    image = draw_floor(ax, data.floor.history[timestamp], cmap)

    _ = draw_roi(ax, roi.x, roi.y, floor_config.roi_size)

    ax.set_title(f"ROI at frame {(data.frame_number - len(axs.flatten()) + 1) + timestamp}")
if image:
    plt.tight_layout(rect=(0, 0, 0.9, 1))
    cbar_ax = fig.add_axes((0.91, 0.15, 0.02, 0.7))
    fig.colorbar(image, cax=cbar_ax)
else:
    plt.tight_layout()

print(data.frame_number)
plt.show()
