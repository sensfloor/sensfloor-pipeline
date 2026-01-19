from typing import Protocol


class MessageValidator(Protocol):
    def __call__(self, message: dict) -> bool: ...


class PositionValidator(MessageValidator):
    def __init__(self, min_x: int, max_x: int, min_y: int, max_y: int) -> None:
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def __call__(self, message: dict) -> bool:
        return self.min_x <= message["x"] <= self.max_x and self.min_y <= message["y"] <= self.max_y
