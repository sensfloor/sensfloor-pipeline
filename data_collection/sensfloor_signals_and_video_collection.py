import csv
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

import cv2
from serial import Serial

START_BYTE = 0xFD
MESSAGE_LENGTH = 17
SERIAL_PORT = "/dev/tty.usbserial-A571V1ZF"
READOUT_DIR = "data"
FPS = 15


def collect_messages(message_queue: Queue[bytes], stop_event: threading.Event) -> None:
    with Serial(str(SERIAL_PORT), 115200, timeout=1) as ser:
        # Read messages until script terminates
        while not stop_event.is_set():
            # Collect all bytes of a message
            while True:
                byte = ser.read(size=1)

                if not byte or byte[0] != START_BYTE:
                    continue

                rest = ser.read(MESSAGE_LENGTH - 1)

                if len(rest) == MESSAGE_LENGTH - 1:
                    message_queue.put(byte + rest)
                else:
                    pass  # Skip invalid message


def message_to_dict(message: bytearray, frame_number: int, timestamp_ns: int) -> dict[str, Any]:
    message_dict = {
        "timestamp": timestamp_ns,
        "frame_number": frame_number,
        "group_id": int.from_bytes(message[1:3], byteorder="big"),
        "magic_number": 23,
        "x": message[3],
        "y": message[4],
    }
    field_values = message[9:]
    message_dict.update({str(i): field_values[i] for i in range(len(field_values))})
    return message_dict


class CollectMessageThreadHandle(NamedTuple):
    message_queue: Queue
    stop_event: threading.Event
    thread: threading.Thread


def start_collect_messages_thread() -> CollectMessageThreadHandle:
    message_queue = Queue()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=collect_messages,
        kwargs={"message_queue": message_queue, "stop_event": stop_event},
    )
    thread.start()
    return CollectMessageThreadHandle(message_queue, stop_event, thread)


def stop_collect_messages_thread(handle: CollectMessageThreadHandle) -> None:
    handle.stop_event.set()
    handle.thread.join()


def read_message_from_queue(message_queue: Queue, frame_number: int, timestamp_ns: int) -> Iterator[dict]:
    while not message_queue.empty():
        message = message_queue.get()
        message_dict = message_to_dict(message, frame_number, timestamp_ns)
        yield message_dict


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
