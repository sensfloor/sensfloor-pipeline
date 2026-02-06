import time
from typing import Any

from serial import Serial

from .base_messages_provider import BaseMessagesProvider

START_BYTE = 0xFD
MESSAGE_LENGTH = 17
VALID_MESSAGE_IDENTIFIER = 0x11


def message_to_dict(message: bytearray, timestamp_ns: int) -> dict[str, Any]:
    return {
        "timestamp": timestamp_ns,
        "group_id": int.from_bytes(message[1:3], byteorder="big"),
        "magic_number": 23,
        "x": message[3],
        "y": message[4],
        "0": message[9],
        "1": message[10],
        "2": message[11],
        "3": message[12],
        "4": message[13],
        "5": message[14],
        "6": message[15],
        "7": message[16],
    }


class SerialMessagesProvider(BaseMessagesProvider):
    def __init__(
        self,
        serial_port: str,
        baudrate: int = 115200,
    ) -> None:
        super().__init__()
        self.serial_port = serial_port
        self.baudrate = baudrate
        self._start_time_ns = 0

    def _run(self) -> None:
        self._start_time_ns = time.perf_counter_ns()
        with Serial(self.serial_port, self.baudrate, timeout=1) as ser:
            # Read messages until script terminates
            while not self._stop_event.is_set():
                message_bytes = bytearray()
                start_byte_found = False

                while True:
                    byte = ser.read(size=1)

                    if not byte:
                        break

                    byte_value = byte[0]

                    if not start_byte_found:
                        if byte_value == START_BYTE:
                            start_byte_found = True
                            message_bytes.append(byte_value)
                    else:
                        message_bytes.append(byte_value)
                        if len(message_bytes) == MESSAGE_LENGTH:
                            # Add message to queue
                            if message_bytes[7] == VALID_MESSAGE_IDENTIFIER:
                                message_dict = message_to_dict(
                                    message_bytes,
                                    int(time.perf_counter_ns() - self._start_time_ns),
                                )
                                self.messages_queue.put(message_dict)
                            # Continue with new message
                            break
