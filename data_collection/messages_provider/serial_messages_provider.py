import time
from typing import Any

from serial import Serial

from .base_messages_provider import BaseMessagesProvider

START_BYTE = 0xFD
MESSAGE_LENGTH = 17


def message_to_dict(message: bytes, timestamp_ns: int) -> dict[str, Any]:
    return {
        "timestamp": timestamp_ns,
        "group_id": int.from_bytes(message[1:3], byteorder="big"),
        "magic_number": 23,
        "x": message[3],
        "y": message[4],
        "0": int(message[9]),
        "1": int(message[10]),
        "2": int(message[11]),
        "3": int(message[12]),
        "4": int(message[13]),
        "5": int(message[14]),
        "6": int(message[15]),
        "7": int(message[16]),
    }


class SerialMessagesProvider(BaseMessagesProvider):
    def __init__(self, serial_port: str, baudrate: int = 115200) -> None:
        super().__init__()
        self.serial_port = serial_port
        self.baudrate = baudrate
        self._start_time_ns = 0

    def _run(self) -> None:
        self._start_time_ns = time.perf_counter_ns()
        with Serial(self.serial_port, self.baudrate, timeout=1) as ser:
            # Read messages until script terminates
            while not self._stop_event.is_set():
                # Collect all bytes of a message
                while True:
                    byte = ser.read(size=1)

                    if not byte or byte[0] != START_BYTE:
                        continue

                    rest = ser.read(MESSAGE_LENGTH - 1)

                    if len(rest) == MESSAGE_LENGTH - 1:
                        message_bytes = byte + rest
                        self.messages_queue.put(
                            message_to_dict(message_bytes, int(time.perf_counter_ns() - self._start_time_ns)),
                        )
                    else:
                        pass  # Skip invalid message
