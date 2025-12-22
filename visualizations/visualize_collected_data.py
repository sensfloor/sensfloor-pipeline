from pathlib import Path

import cv2
import pandas as pd

from data_loading.roi_floor import RoIFloor, RoIFloorConfig

DATA_DIR_PATH = Path("data/2025-12-16_20-45-38-line-felix")
READOUT_PATH = DATA_DIR_PATH / "sensfloor_readout.csv"
VIDEO_PATH = DATA_DIR_PATH / "video.mp4"

SIGNAL_THRESHOLD = 145

readout_df = pd.read_csv(READOUT_PATH)

display_w = 400
display_h = 600

cv2.namedWindow("Floor Heatmap", cv2.WINDOW_NORMAL)
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Floor Heatmap", display_w, display_h)
cv2.resizeWindow("Video", 640, 480)

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)

frame_number = 0
floor_config = RoIFloorConfig(x_size=6, y_size=4, history_maxlen=10, roi_size=3)
floor = RoIFloor(floor_config)

while True:
    ret, frame = cap.read()
    messages = readout_df[readout_df["frame_number"] == frame_number]
    positions = messages[["x", "y"]].to_numpy() - 1
    signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
    signals[signals < SIGNAL_THRESHOLD] = 0
    floor.update(positions=positions, signals=signals)
    current_floor = floor.history[-1].astype("uint8")
    roi = floor.get_roi()
    current_floor[current_floor < floor_config.active_field_min_value] = 0

    display_img = cv2.cvtColor(current_floor, cv2.COLOR_GRAY2BGR)
    if roi is not None:
        x = roi.x * 4
        y = roi.y * 4
        size = floor.roi_size

        cv2.rectangle(
            display_img,
            (y, x),
            (y + size * 4 - 1, x + size * 4 - 1),
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
