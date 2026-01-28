import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.ndimage import center_of_mass, label, sum_labels

from data_loading.roi_floor import RoIFloor, RoIFloorConfig
from data_loading.roi_offset_strategy import get_exhaustive_offsets


class SensfloorKalmanFilter:
    def __init__(self, fps: float) -> None:
        self.dt = 1 / fps
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.last_x = None
        self.reset()

    def reset(self) -> None:
        # TODO: Test usage of last position as initial belief if reset due to floor exit
        self.kf.P = np.eye(4) * 10
        self.kf.x = np.array([0, 0, 0, 0], dtype=np.float64)  # (Initial) State estimate
        self.kf.F = np.array(  # State transition matrix
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.kf.Q = np.eye(4) * 0.0003  # Transition noise
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)  # State -> Measurement mapping
        self.kf.R = np.eye(2) * 0.01  # Measurement noise

    def predict(self) -> None:
        self.kf.predict()

    def update(self, position: np.ndarray) -> None:
        self.last_x = self.x
        self.kf.update(position)

    @property
    def x(self) -> np.ndarray:
        return self.kf.x[:2]


def calculate_activation_cluster_means(floor_activations: np.ndarray, idle_field_value: int) -> np.ndarray:
    mask = floor_activations > idle_field_value

    labeled_array, num_features = label(mask)  # type: ignore

    if num_features == 0:
        return np.array([])

    clusters = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
    means = np.array([cluster.mean(axis=0) for cluster in clusters])

    labels = np.arange(1, num_features + 1)
    means = np.array(center_of_mass(mask, labeled_array, labels))

    field_values = sum_labels(floor_activations, labeled_array, labels)
    highest_means = means[np.argsort(field_values)[::-1]][:2]

    if len(highest_means) > 1:
        return highest_means.mean(axis=0).flatten()
    return highest_means.flatten()


RESET_FILTER_THRESHOLD = 10
VIDEO_WINNAME = "Video"
COMBINED_WINNAME = "Raw position signal vs Kalman filtered"


def main(data_dir: Path, signal_threshold: int) -> None:  # noqa: PLR0915
    readout_path = data_dir / "sensfloor_readout.csv"
    video_path = data_dir / "video.mp4"

    readout_df = pd.read_csv(readout_path)
    readout_lookup = dict(list(readout_df.groupby("frame_number")))

    display_w, display_h = 400, 600

    cv2.namedWindow(VIDEO_WINNAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(COMBINED_WINNAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(VIDEO_WINNAME, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(VIDEO_WINNAME, 640, 480)

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

    kf = SensfloorKalmanFilter(fps)
    frame_number = 0
    frames_without_signal = 0
    while True:
        ret, frame = cap.read()
        messages = readout_lookup.get(frame_number, pd.DataFrame())

        if not messages.empty:
            positions = messages[["x", "y"]].to_numpy() - 1
            signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
            floor.update(positions=positions, signals=signals)

        current_floor = floor.history[-1].astype("uint8")

        scale_h = display_h / current_floor.shape[0]
        scale_w = display_w / current_floor.shape[1]

        img_raw = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        img_kf = np.zeros((display_h, display_w, 3), dtype=np.uint8)

        kf.predict()

        position = calculate_activation_cluster_means(current_floor, floor_config.idle_field_value)

        if len(position) > 0:
            # Draw raw signal position
            r, c = position
            cv2.drawMarker(img_raw, (int(c * scale_w), int(r * scale_h)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # KF processing
            if frames_without_signal > RESET_FILTER_THRESHOLD:
                kf.reset()
            frames_without_signal = 0
            kf.update(position)
        else:
            frames_without_signal += 1

        # Draw KF position
        kf_r, kf_c = kf.x
        cv2.drawMarker(img_kf, (int(kf_c * scale_w), int(kf_r * scale_h)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Arrange comparison window
        header_h = 40
        divider_width = 10
        divider = np.ones((display_h, divider_width, 3), dtype=np.uint8) * 255
        header = np.ones((header_h, display_w * 2 + divider_width, 3), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(header, "RAW SENSOR DATA", (10, 25), font, 1, (0, 0, 0), 1)
        cv2.putText(header, "KALMAN FILTER DATA", (display_w + 10, 25), font, 1, (0, 0, 0), 1)

        combined_floor = np.hstack((img_raw, divider, img_kf))
        final_view = np.vstack((header, combined_floor))

        cv2.imshow(COMBINED_WINNAME, final_view)
        cv2.imshow(VIDEO_WINNAME, frame)

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
