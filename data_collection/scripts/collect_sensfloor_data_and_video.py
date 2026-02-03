import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

from data_collection.messages_provider.serial_messages_provider import SerialMessagesProvider


def main(readout_dir: Path, serial_port: str, fps: int) -> None:
    recording_datetime = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    base_path = Path(readout_dir) / recording_datetime
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
        fps,
        (frame_width, frame_height),
    )
    frame_interval_length = 1.0 / fps

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
    messages_provider = SerialMessagesProvider(serial_port)
    messages_provider.start()
    recording_start_time = time.perf_counter()
    try:
        frame_number = 0

        while cap.isOpened():
            frame_start_time = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                break

            # Collect all messages for current frame from queue
            messages = messages_provider.get_messages()
            for message in messages:
                message["frame_number"] = frame_number
                writer.writerow(message)

            # Display recorded video frame
            out.write(frame)
            cv2.imshow("Recording...", frame)

            # Check whether stop data collection
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Wait until next frame
            frame_duration = time.perf_counter() - frame_start_time
            sleep_time = frame_interval_length - frame_duration

            frame_number += 1

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(
                    f"Processing took longer than {frame_interval_length:.4f}s "
                    f"(took {frame_duration:.4f}s). Video is not {fps} FPS!",
                )

    finally:
        # Print recording duration
        recording_time = time.perf_counter() - recording_start_time
        print(f"Recorded for {recording_time}s")

        # Stop provider
        messages_provider.stop()

        # Close open files
        readout_file.close()

        # Close video stream
        cap.release()
        out.release()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect data from sensfloor and record video synchronously.",
    )

    parser.add_argument(
        "--readout-dir",
        type=Path,
        default=Path("data"),
        help="Root folder for recorded data.",
    )

    parser.add_argument(
        "--serial-port",
        type=str,
        required=True,
        help="Serial port of sensfloor connector.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Framerate of recorded data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.readout_dir, args.serial_port, args.fps)
