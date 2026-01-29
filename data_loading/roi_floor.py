from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loading.floor import PATCH_SIZE, Floor, FloorConfig
from data_loading.roi_offset_strategy import OffsetStrategy, get_offset_strategy


@dataclass(frozen=True)
class RoI:
    x: int
    y: int
    history: np.ndarray


@dataclass(kw_only=True, frozen=True)
class RoIFloorConfig(FloorConfig):
    roi_size: int | None
    offset_strategy: OffsetStrategy = OffsetStrategy.CENTER


def get_relevant_messages_mask(signals: np.ndarray, active_field_min_value: int) -> np.ndarray:
    return np.where(signals >= active_field_min_value)[0]


class RoIFloor(Floor):
    def __init__(self, config: RoIFloorConfig) -> None:
        if config.roi_size is not None and (config.x_size < config.roi_size or config.y_size < config.roi_size):
            message = "RoI too large for floor size"
            raise ValueError(message)

        super().__init__(config)
        self.roi_size = config.roi_size
        self.last_updated_positions: np.ndarray | None = None

        self.x_y_corner_offsets: list[tuple[int, int]] = []
        if self.roi_size is not None:
            self.x_y_corner_offsets = get_offset_strategy(config.offset_strategy)(self.roi_size)

    def update(self, positions: np.ndarray, signals: np.ndarray) -> None:
        relevant_messages_mask = get_relevant_messages_mask(signals, self.config.active_field_min_value)
        self.last_updated_positions = positions[relevant_messages_mask]  # Save only active field positions
        super().update(positions, signals)

    def get_roi(self) -> None | RoI:
        # Return no region of interest if no updates exist
        if self.last_updated_positions is None:
            return None

        if self.roi_size is None:
            return RoI(x=0, y=0, history=self.history)

        unique_roi_positions = self.get_unique_roi_positions()

        max_signal_sum = 0
        roi = None
        for x_roi_top_left, y_roi_top_left in unique_roi_positions:
            roi_range = self.roi_size * PATCH_SIZE
            roi_history = self.history[
                :,
                x_roi_top_left : x_roi_top_left + roi_range,
                y_roi_top_left : y_roi_top_left + roi_range,
            ]

            signal_sum = roi_history.sum()
            if signal_sum > max_signal_sum:
                max_signal_sum = signal_sum
                roi = RoI(x_roi_top_left // PATCH_SIZE, y_roi_top_left // PATCH_SIZE, roi_history)

        return roi

    def get_unique_roi_positions(self) -> set[tuple[int, int]]:
        if self.last_updated_positions is None:
            return set()

        unique_positions = set()
        for x, y in self.last_updated_positions:
            for x_offset, y_offset in self.x_y_corner_offsets:
                x_roi_top_left = self._get_clamped_top_left(x, x_offset, axis=0)
                y_roi_top_left = self._get_clamped_top_left(y, y_offset, axis=1)
                unique_positions.add((x_roi_top_left, y_roi_top_left))
        return unique_positions

    def _get_clamped_top_left(self, coordinate: int, offset: int, axis: int) -> int:
        if self.roi_size is None:
            message = (
                "Cannot calculate ROI coordinates when roi_size is None. "
                "This method should only be called when a specific ROI size is configured."
            )
            raise RuntimeError(message)

        top_left = coordinate - offset

        if top_left <= 0:
            return 0

        if (top_left + self.roi_size) * PATCH_SIZE > self.patches.shape[axis]:
            return self.patches.shape[axis] - self.roi_size * PATCH_SIZE

        return top_left * PATCH_SIZE


def create_roi_floor(config: RoIFloorConfig, sensfloor_readout: pd.DataFrame, frame_number: int) -> RoIFloor:
    floor = RoIFloor(config)
    earliest_frame_included = frame_number - config.history_maxlen + 1
    for current_frame_number in range(earliest_frame_included, frame_number + 1):
        messages = sensfloor_readout[sensfloor_readout["frame_number"] == current_frame_number]
        positions = messages[["x", "y"]].to_numpy() - 1  # Subtract 1 from positions because they start at 1,1
        signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
        floor.update(positions, signals)
    return floor
