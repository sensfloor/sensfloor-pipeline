import csv
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

from data_collection.sensfloor_signals_and_video_collection import (
    read_message_from_queue,
    start_collect_messages_thread,
    stop_collect_messages_thread,
)

READOUT_DIR = "data"
FPS = 15  # TODO: Argument


def main() -> None:
    recording_datetime = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    base_path = Path(READOUT_DIR) / recording_datetime
    recording_csv_path = base_path / "sensfloor_readout.csv"
    recording_video_path = base_path / "video.mp4"

    # Create readout directory
    base_path.mkdir(parents=True, exist_ok=True)

    # Initialize camera feed
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        str(recording_video_path),
        fourcc,
        FPS,
        (frame_width, frame_height),
    )
    frame_interval_length = 1.0 / FPS

    # Initialize file readout
    readout_file = recording_csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "timestamp",
        "frame_number",
        "group_id",
        "magic_number",
        "x",
        "y",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ]
    writer = csv.DictWriter(readout_file, fieldnames=fieldnames)
    writer.writeheader()

    # Start messages thread that writes messages in queue
    collect_messages_handle = start_collect_messages_thread()
    recording_start_time = time.perf_counter()
    try:
        frame_number = 0

        while cap.isOpened():
            frame_start_time = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                break

            # Collect all messages for current frame from queue
            timestamp_ns = int(time.perf_counter_ns() - recording_start_time)

            for message_dict in read_message_from_queue(
                collect_messages_handle.message_queue,
                frame_number,
                timestamp_ns,
            ):
                writer.writerow(message_dict)

            out.write(frame)
            cv2.imshow("Recording...", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_duration = time.perf_counter() - frame_start_time
            sleep_time = frame_interval_length - frame_duration

            frame_number += 1

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(
                    f"Processing took longer than {frame_interval_length:.4f}s "
                    f"(took {frame_duration:.4f}s). Video is not {FPS} FPS!",
                )

    finally:
        recording_time = time.perf_counter() - recording_start_time
        print(f"Recorded for {recording_time}s")
        stop_collect_messages_thread(collect_messages_handle)

        # Close open files
        readout_file.close()

        # Close video stream
        cap.release()
        out.release()


if __name__ == "__main__":
    main()
