import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data_loading.roi_floor import RoIFloor, RoIFloorConfig
from data_loading.roi_offset_strategy import get_exhaustive_offsets
from tracking.clustering import calculate_activation_cluster_means
from tracking.person_tracker import PersonTracker

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

    person_tracker = PersonTracker(fps, floor_config.idle_field_value, filter_reset_threshold=10)
    frame_number = 0
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

        # Calculate and draw raw signal position
        position = calculate_activation_cluster_means(current_floor, floor_config.idle_field_value)
        if len(position) > 0:
            row, column = position
            cv2.drawMarker(img_raw, (int(column * scale_w), int(row * scale_h)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Calculate and draw KF position
        kf_r, kf_c = person_tracker.track(current_floor)
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
