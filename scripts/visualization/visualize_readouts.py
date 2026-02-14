import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def visualize_read_outs(csv_path: Path) -> None:
    floor_readout = pd.read_csv(csv_path)
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
    plt.title(f"Cumulative Activity Heatmap: {csv_path.parent.name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize readout heatmap")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to sensfloor readout to visualize.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_read_outs(args.csv)
