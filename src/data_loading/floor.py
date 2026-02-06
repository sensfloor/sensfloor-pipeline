from collections import deque
from dataclasses import dataclass

import numpy as np


def interpolate_signal(signal: np.ndarray) -> np.ndarray:
    averages = signal.reshape((4, 2)).mean(axis=1)
    return np.array(
        [
            [averages[2], signal[5], signal[6], averages[3]],
            [signal[4], averages[2], averages[3], signal[7]],
            [signal[3], averages[1], averages[0], signal[0]],
            [averages[1], signal[2], signal[1], averages[0]],
        ],
    )


@dataclass(kw_only=True, frozen=True)
class FloorConfig:
    x_size: int
    y_size: int
    history_maxlen: int
    idle_field_value: int = 127
    active_field_min_value: int = 140
    remove_noise: bool = False


PATCH_SIZE = 4


class Floor:
    """
    Class that models a x by y patches SensFloor with 8 fields per patch.

    **Field value ranges**:
    - [0, active_field_value[: Mapped to idle_field_value
    - [active_field_value, 255]: Movement detected -> Signal value untouched
    """

    def __init__(self, config: FloorConfig) -> None:
        self.config = config
        self.history_queue = deque(maxlen=config.history_maxlen)
        self.updated_positions_history = deque(maxlen=1)
        self.patches = np.ones((config.x_size * PATCH_SIZE, config.y_size * PATCH_SIZE)) * config.idle_field_value

    @property
    def shape(self) -> tuple[int, int]:
        return self.config.x_size, self.config.y_size

    @property
    def history(self) -> np.ndarray:
        empty_history = self.config.history_maxlen - len(self.history_queue)
        zero_fill = (
            np.ones((empty_history, self.config.x_size * PATCH_SIZE, self.config.y_size * PATCH_SIZE))
            * self.config.idle_field_value
        )
        return np.stack([*zero_fill, *list(self.history_queue)])

    def update(self, positions: np.ndarray, signals: np.ndarray) -> None:
        if self.config.remove_noise:
            self.remove_noise()

        cleaned_signal = np.where(signals < self.config.active_field_min_value, self.config.idle_field_value, signals)
        for (x, y), signal in zip(positions, cleaned_signal, strict=True):
            interpolated_signal = interpolate_signal(signal)
            x_patches = x * PATCH_SIZE
            y_patches = y * PATCH_SIZE
            self.patches[x_patches : x_patches + PATCH_SIZE, y_patches : y_patches + PATCH_SIZE] = interpolated_signal

        self.history_queue.append(self.patches.copy())
        self.updated_positions_history.append(positions.tolist())

    def remove_noise(self) -> None:
        updated_positions = list(self.updated_positions_history)
        unique_positions = {tuple(item) for timestep in updated_positions if timestep for item in timestep}

        keep_mask = np.ones(self.patches.shape, dtype=bool)  # 1 => True

        for x, y in unique_positions:
            x_start, y_start = x * PATCH_SIZE, y * PATCH_SIZE
            keep_mask[x_start : x_start + PATCH_SIZE, y_start : y_start + PATCH_SIZE] = False

        self.patches[keep_mask] = self.config.idle_field_value
