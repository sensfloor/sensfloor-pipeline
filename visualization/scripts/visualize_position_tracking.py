import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, label, sum_labels

from data_loading.roi_floor import RoIFloor, RoIFloorConfig
from data_loading.roi_offset_strategy import get_exhaustive_offsets


def calculate_activation_cluster_means(floor_activations: np.ndarray, idle_field_value: int) -> np.ndarray:
    mask = floor_activations > idle_field_value

    labeled_array, num_features = label(mask)  # type: ignore  # noqa: PGH003

    if num_features == 0:
        return np.array([])

    clusters = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
    means = np.array([cluster.mean(axis=0) for cluster in clusters])

    labels = np.arange(1, num_features + 1)
    means = np.array(center_of_mass(mask, labeled_array, labels))

    field_values = sum_labels(floor_activations, labeled_array, labels)
    highest_means = means[np.argsort(field_values)[::-1]][:2]

    if len(highest_means) == 2:
        return highest_means.mean(axis=0).flatten()
    return highest_means.flatten()


def main(data_dir: Path, signal_threshold: int) -> None:
    readout_path = data_dir / "sensfloor_readout.csv"
    video_path = data_dir / "video.mp4"

    readout_df = pd.read_csv(readout_path)
    readout_lookup = dict(list(readout_df.groupby("frame_number")))

    display_w, display_h = 400, 600

    cv2.namedWindow("Floor Heatmap", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Floor Heatmap", display_w, display_h)
    cv2.resizeWindow("Video", 640, 480)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps)

    floor_config = RoIFloorConfig(
        x_size=6,
        y_size=4,
        history_maxlen=10,
        roi_size=3,
        active_field_min_value=signal_threshold,
        offset_strategy=get_exhaustive_offsets,
        remove_noise=True,
    )
    floor = RoIFloor(floor_config)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        messages = readout_lookup.get(frame_number, pd.DataFrame())

        if not messages.empty:
            positions = messages[["x", "y"]].to_numpy() - 1
            signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
            floor.update(positions=positions, signals=signals)

        current_floor = floor.history[-1].astype("uint8")

        heatmap_resized = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        scale_h = display_h / current_floor.shape[0]
        scale_w = display_w / current_floor.shape[1]

        position = calculate_activation_cluster_means(current_floor, floor_config.idle_field_value)

        if len(position) == 2:
            row, col = position
            draw_x = int(col * scale_w)
            draw_y = int(row * scale_h)

            cv2.drawMarker(heatmap_resized, (draw_x, draw_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(heatmap_resized, (draw_x, draw_y), 5, (0, 255, 0), -1)

        cv2.imshow("Floor Heatmap", heatmap_resized)
        cv2.imshow("Video", frame)

        frame_number += 1
        if not ret:
            break

        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize collected video and readout")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Required path to data directory containing video and sensfloor readout.",
    )
    parser.add_argument(
        "--signal-threshold",
        type=int,
        default=145,
        help="Threshold value for signal being not considered as noise",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.signal_threshold)
