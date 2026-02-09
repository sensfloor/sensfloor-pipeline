import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.data_loading.floor import PATCH_SIZE
from src.data_loading.roi_floor import RoIFloor, RoIFloorConfig
from src.data_loading.roi_offset_strategy import OffsetStrategy
from src.definitions import READOUT_FILENAME, VIDEO_FILENAME


def main(data_dir: Path, signal_threshold: int) -> None:
    readout_path = data_dir / READOUT_FILENAME
    video_path = data_dir / VIDEO_FILENAME

    readout_df = pd.read_csv(readout_path)

    display_w = 400
    display_h = 600

    cv2.namedWindow("Floor Heatmap", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Floor Heatmap", display_w, display_h)
    cv2.resizeWindow("Video", 640, 480)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps)

    frame_number = 0
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

    while True:
        ret, frame = cap.read()
        messages = readout_df[readout_df["frame_number"] == frame_number]
        positions = messages[["x", "y"]].to_numpy() - 1
        signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
        floor.update(positions=positions, signals=signals)
        current_floor = floor.history[-1].astype("uint8")
        roi = floor.get_roi()
        current_floor = floor.history[-1].astype("float32")

        current_floor = (current_floor - 127.0) / (255.0 - 127.0)
        current_floor = current_floor.clip(0.0, 1.0)
        current_floor_u8 = (current_floor * 255.0).astype("uint8")

        display_img = cv2.cvtColor(current_floor_u8, cv2.COLOR_GRAY2BGR)
        if roi is not None:
            x = roi.x * PATCH_SIZE
            y = roi.y * PATCH_SIZE
            roi_size = floor.roi_size

            assert roi_size is not None, "ROI size must be specified for visualization"  # noqa: S101

            cv2.rectangle(
                display_img,
                (y, x),
                (y + roi_size * PATCH_SIZE - 1, x + roi_size * PATCH_SIZE - 1),
                (0, 0, 255),
                1,
            )

        heatmap_resized = cv2.resize(display_img, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Floor Heatmap", heatmap_resized)

        if not ret:
            break

        cv2.imshow("Video", frame)
        frame_number += 1
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
