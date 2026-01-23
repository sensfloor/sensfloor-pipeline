from collections.abc import Callable

OffsetStrategy = Callable[[int], list[tuple[int, int]]]
"""Callable type to generate ROI offsets centered around the activated sensor position.

Args:
    roi_size: The size of the squared region of interest.

Returns:
    A list of (x_offset, y_offset) tuples representing possible positions
    of the ROI's top-left corner relative to an activated sensor.
"""

def get_center_offsets(roi_size: int) -> list[tuple[int, int]]:
    half = roi_size // 2
    if roi_size % 2 == 1:
        return [(half, half)]
    offsets = [half, half - 1]
    return [(x, y) for x in offsets for y in offsets]


def get_exhaustive_offsets(roi_size: int) -> list[tuple[int, int]]:
    return [(x_offset, y_offset) for x_offset in range(roi_size) for y_offset in range(roi_size)]
