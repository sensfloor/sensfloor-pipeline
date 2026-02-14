from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.definitions import HOLD_OUT_DATA_PATH, READOUT_FILENAME, TRAIN_DATA_PATH


def visualize_read_outs(data_path: Path) -> None:
    floor_readout = pd.read_csv(data_path / READOUT_FILENAME)
    floor_readout = floor_readout.drop(columns=["timestamp", "frame_number", "group_id", "magic_number"])

    sensor_columns = ["0", "1", "2", "3", "4", "5", "6", "7"]
    floor_readout["sensor_val"] = floor_readout[sensor_columns].mean(axis=1)
    floor_readout["patch"] = floor_readout["sensor_val"] - 127
    floor_readout["patch"] = floor_readout["patch"].clip(lower=0)
    grid_groups = floor_readout.groupby(["x", "y"])["patch"].sum()

    heatmap_data = grid_groups.unstack(level="x").fillna(0)

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap="inferno", origin="lower")
    plt.colorbar(label="Total Accumulated Activity (Intensity * Frequency)")
    plt.title(f"Cumulative Activity Heatmap: {data_path.name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


if __name__ == "__main__":
    training_folders = [directory for directory in TRAIN_DATA_PATH.iterdir() if directory.is_dir()]
    hold_out_folders = [directory for directory in HOLD_OUT_DATA_PATH.iterdir() if directory.is_dir()]
    all_folders = training_folders + hold_out_folders

    for folder in all_folders:
        visualize_read_outs(folder)
