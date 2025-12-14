import csv
import threading
import time
from csv import DictWriter
from datetime import datetime
from pathlib import Path
from queue import Queue
from zoneinfo import ZoneInfo

import cv2
from serial import Serial

SERIAL_PORT = "/dev/tty.usbserial-A571V1ZF"
READOUT_DIR = "data"
FPS = 15

recording_datetime = datetime.now(ZoneInfo("Europe/Berlin")).strftime(
    "%Y-%m-%d_%H-%M-%S",
)
base_path = Path(READOUT_DIR) / recording_datetime
recording_csv_path = base_path / "sensfloor_readout.csv"
recording_video_path = base_path / "video.mp4"

recording_start_ns = time.perf_counter_ns()

message_queue = Queue()
stop_event = threading.Event()


def read_messages() -> None:
    with Serial(str(SERIAL_PORT), 115200, timeout=1) as ser:
        # Read messages until script terminates
        while not stop_event.is_set():
            message = bytearray()
            start_byte_found = False

            # Collect all bytes of a message
            while True:
                byte = ser.read(size=1)

                if not byte:
                    break

                byte_value = byte[0]

                if not start_byte_found:
                    if byte_value == 0xFD:
                        start_byte_found = True
                        message.append(byte_value)
                else:
                    message.append(byte_value)
                    if len(message) == 17:
                        break

            if len(message) == 17:
                message_queue.put(message)


def write_message_in_file(
    message: bytearray,
    frame_number: int,
    writer: DictWriter,
) -> None:
    hex_arr = list(message)

    # Check for invalid message?
    if hex_arr[7] != 0x11:
        return

    row = {
        "timestamp": str(int(time.perf_counter_ns() - recording_start_ns)),
        "frame_number": frame_number,
        "group_id": int.from_bytes(hex_arr[1:3], byteorder="big"),
        "magic_number": 23,
        "x": int.from_bytes([hex_arr[3]]),
        "y": int.from_bytes([hex_arr[4]]),
        **{str(field_id): sensor_value for field_id, sensor_value in enumerate(hex_arr[9:])},
    }

    writer.writerow(row)


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
messages_thread = threading.Thread(target=read_messages)
messages_thread.start()

recording_start_time = time.perf_counter()
try:
    frame_number = 0

    while cap.isOpened():
        frame_start_time = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            break

        # Collect all messages for current frame from queue
        while not message_queue.empty():
            message = message_queue.get()
            write_message_in_file(message, frame_number, writer)

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
                f"Processing took longer than {frame_interval_length:.4f}s (took {frame_duration:.4f}s). Video is not {FPS} FPS!",
            )

finally:
    recording_time = time.perf_counter() - recording_start_time
    print(f"Recorded for {recording_time}s")
    stop_event.set()
    messages_thread.join()

    # Close open files
    readout_file.close()

    # Close video stream
    cap.release()
    out.release()
