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

        max_signal_sum = 0
        roi = None
        for x, y in self.last_updated_positions:
            x_roi_top_left = self._get_roi_x_top_left(x)
            y_roi_top_left = self._get_roi_y_top_left(y)
            roi_range = self.roi_size * 4
            roi_history = self.history[
                :,
                x_roi_top_left : x_roi_top_left + roi_range,
                y_roi_top_left : y_roi_top_left + roi_range,
            ]

            signal_sum = roi_history[-1].sum()
            if signal_sum > max_signal_sum:
                max_signal_sum = signal_sum
                roi = RoI(x_roi_top_left // 4, y_roi_top_left // 4, roi_history)

        return roi

    def _get_roi_x_top_left(self, x: int) -> int:
        x_top_left = x - (self.roi_size // 2)

        if x_top_left <= 0:
            return 0

        if (x_top_left + self.roi_size) * 4 > self.patches.shape[0]:
            return self.patches.shape[0] - self.roi_size * 4

        return x_top_left * 4

    def _get_roi_y_top_left(self, y: int) -> int:
        y_top_left = y - (self.roi_size // 2)

        if y_top_left <= 0:
            return 0

        if (y_top_left + self.roi_size) * 4 > self.patches.shape[1]:
            return self.patches.shape[1] - self.roi_size * 4

        return y_top_left * 4
