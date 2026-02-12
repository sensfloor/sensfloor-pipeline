import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data_loading.roi_floor import RoIFloor, RoIFloorConfig
from src.data_loading.roi_offset_strategy import OffsetStrategy
from src.tracking.clustering import calculate_activation_cluster_means
from src.tracking.person_tracker import PersonTracker


def plot_paths(raw_signals_path: np.ndarray, kalman_filter_path: np.ndarray) -> None:
    plt.rcParams.update(
        {
            "font.family": "Courier New",
            "font.size": 15,
        },
    )

    _, axs = plt.subplots(ncols=2, figsize=(8, 6))

    paths = [
        ("Raw Signal Trajectory", raw_signals_path),
        ("Kalman Filter path", kalman_filter_path),
    ]

    for ax, (title, path) in zip(axs, paths, strict=True):
        x = path[:-1, 1]
        y = path[:-1, 0]
        u = path[1:, 1] - path[:-1, 1]
        v = path[1:, 0] - path[:-1, 0]

        ax.quiver(
            x,
            y,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.004,
            headwidth=7,
            headlength=7,
            headaxislength=7,
        )

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 6.5)

        # Paint floor grid
        for x in range(5):
            ax.vlines(x, 0, 6, colors="0.7", linewidth=1, zorder=0, alpha=0.5)
        for y in range(7):
            ax.hlines(y, 0, 4, colors="0.7", linewidth=1, zorder=0, alpha=0.5)

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
    plt.tight_layout()

    save_path = Path("outputs/images/filter_visualization.svg")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Saved plot to {save_path}")


def main(readout_path: Path, signal_threshold: int, start_time_sec: int, end_time_sec: int, fps: int) -> None:
    start_frame = start_time_sec * fps
    end_frame = end_time_sec * fps

    readout_df = pd.read_csv(readout_path)
    readout_df = readout_df[(readout_df["frame_number"] >= start_frame) & (readout_df["frame_number"] <= end_frame)]
    readout_lookup = dict(list(readout_df.groupby("frame_number")))

    floor_config = RoIFloorConfig(
        x_size=6,
        y_size=4,
        history_maxlen=10,
        roi_size=3,
        active_field_min_value=signal_threshold,
        offset_strategy=OffsetStrategy.EXHAUSTIVE,
        remove_noise=True,
    )
    floor = RoIFloor(floor_config)

    person_tracker = PersonTracker(fps, filter_reset_threshold=10, floor_config=floor_config)

    raw_signal_positions = []
    kalman_filter_positions = []
    for messages in readout_lookup.values():
        positions = np.array([])
        signals = np.array([])
        if not messages.empty:
            positions = messages[["x", "y"]].to_numpy() - 1
            signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
            floor.update(positions=positions, signals=signals)

        current_floor = floor.history[-1].astype("uint8")

        position = calculate_activation_cluster_means(current_floor, floor_config.idle_field_value)
        if len(position) > 0:
            raw_signal_positions.append(position / 4)

        kalman_filter_positions.append(person_tracker.track(positions, signals))

    raw_signals_path = np.array(raw_signal_positions)
    kalman_filter_path = np.array(kalman_filter_positions)

    print(f"Unique raw signals: {len(np.unique(raw_signals_path, axis=0))}")
    print(f"Unique kalman filter signals: {len(np.unique(kalman_filter_path, axis=0))}")
    plot_paths(raw_signals_path, kalman_filter_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize collected video and readout")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Required path to sensfloor readout csv.",
    )
    parser.add_argument(
        "--signal-threshold",
        type=int,
        default=145,
        help="Threshold value for signal being not considered as noise",
    )
    parser.add_argument(
        "--start-time-sec",
        type=int,
        required=True,
        help="Start time in seconds",
    )
    parser.add_argument(
        "--end-time-sec",
        type=int,
        required=True,
        help="End time in seconds",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Video recording FPS",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.csv, args.signal_threshold, args.start_time_sec, args.end_time_sec, args.fps)
