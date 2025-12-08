from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loading.floor import Floor, FloorConfig


@dataclass
class RoI:
    x: int
    y: int
    history: np.ndarray


@dataclass
class RoIFloorConfig(FloorConfig):
    roi_size: int


class RoIFloor(Floor):
    def __init__(self, config: RoIFloorConfig) -> None:
        if config.x_size < config.roi_size or config.y_size < config.roi_size:
            message = "RoI too large for floor size"
            raise ValueError(message)

        super().__init__(config)
        self.roi_size = config.roi_size
        self.last_updated_positions: np.ndarray | None = None

    def update(self, positions: np.ndarray, signals: np.ndarray) -> None:
        self.last_updated_positions = positions
        super().update(positions, signals)

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


def create_roi_floor(config: RoIFloorConfig, sensfloor_readout: pd.DataFrame, frame_number: int) -> RoIFloor:
    floor = RoIFloor(config)

    earliest_frame_included = frame_number - config.history_maxlen
    for current_frame_number in range(earliest_frame_included, frame_number + 1):
        messages = sensfloor_readout[sensfloor_readout["frame_number"] == current_frame_number]
        positions = (
            messages[["x", "y"]].to_numpy() - 1
        ).tolist()  # Subtract 1 from positions because they start at 1,1
        signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
        floor.update(positions, signals)
    return floor
