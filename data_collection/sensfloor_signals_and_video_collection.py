import threading
from collections.abc import Iterator
from queue import Queue
from typing import Any, NamedTuple

from serial import Serial

START_BYTE = 0xFD
MESSAGE_LENGTH = 17
SERIAL_PORT = "/dev/tty.usbserial-A571V1ZF"  # TODO: Argument in script


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
