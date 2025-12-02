from dataclasses import dataclass

import numpy as np

from data_loading.floor import Floor


@dataclass
class RoI:
    x: int
    y: int
    history: np.ndarray


class RoIFloor(Floor):
    def __init__(self, x: int, y: int, history_maxlen: int, roi_size: int) -> None:
        super().__init__(x, y, history_maxlen)
        self.roi_size = roi_size
        self.last_updated_positions: np.ndarray | None = None

    def update(self, positions: list[tuple[int, int]], signals: np.ndarray) -> None:
        self.last_updated_positions = positions
        return super().update(positions, signals)

    def get_roi(self) -> None | RoI:
        # Return no region of interest if no updates exist
        if self.last_updated_positions is None:
            return None

        last_updated_position = self.last_updated_positions[0]
        x = last_updated_position[0]
        y = last_updated_position[1]

        x_kernel = (x - 1) * 4
        y_kernel = (y - 1) * 4
        kernel_range = self.roi_size * 4
        roi_history = self.history[
            :, x_kernel : x_kernel + kernel_range, y_kernel : y_kernel + kernel_range
        ]

        return RoI(x_kernel / 4, y_kernel / 4, roi_history)
