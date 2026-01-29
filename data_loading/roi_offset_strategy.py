from collections.abc import Callable
from enum import Enum


class OffsetStrategy(Enum):
    CENTER = "center"
    EXHAUSTIVE = "exhaustive"


def get_offset_strategy(strategy: OffsetStrategy) -> Callable[[int], list[tuple[int, int]]]:
    if strategy == OffsetStrategy.CENTER:
        return get_center_offsets

    return get_exhaustive_offsets


def get_center_offsets(roi_size: int) -> list[tuple[int, int]]:
    half = roi_size // 2
    if roi_size % 2 == 1:
        return [(half, half)]
    offsets = [half, half - 1]
    return [(x, y) for x in offsets for y in offsets]


def get_exhaustive_offsets(roi_size: int) -> list[tuple[int, int]]:
    return [(x_offset, y_offset) for x_offset in range(roi_size) for y_offset in range(roi_size)]
