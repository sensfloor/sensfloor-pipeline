from collections import deque
from dataclasses import dataclass

import numpy as np


def interpolate_signal(signal: np.ndarray) -> np.ndarray:
    averages = signal.reshape((4, 2)).mean(axis=1)
    return np.array(
        [
            [averages[3], signal[7], signal[0], averages[0]],
            [signal[6], averages[3], averages[0], signal[1]],
            [signal[5], averages[2], averages[1], signal[2]],
            [averages[2], signal[4], signal[3], averages[1]],
        ],
    )


@dataclass
class FloorConfig:
    x_size: int
    y_size: int
    history_maxlen: int


class Floor:
    def __init__(self, config: FloorConfig) -> None:
        self.config = config
        self.history_queue = deque(maxlen=config.history_maxlen)
        self.patches = np.ones((config.x_size * 4, config.y_size * 4)) * 127

    @property
    def shape(self) -> tuple[int, int]:
        return self.config.x_size, self.config.y_size

    @property
    def history(self) -> np.ndarray:
        empty_history = self.config.history_maxlen - len(self.history_queue)
        zero_fill = np.ones((empty_history, self.config.x_size * 4, self.config.y_size * 4)) * 127
        return np.stack([*zero_fill, *list(self.history_queue)])

    def update(self, positions: np.ndarray, signals: np.ndarray) -> None:
        clipped_signals = signals.clip(min=127)
        for (x, y), signal in zip(positions, clipped_signals, strict=True):
            interpolated_signal = interpolate_signal(signal)
            x_patches = x * 4
            y_patches = y * 4
            self.patches[x_patches : x_patches + 4, y_patches : y_patches + 4] = interpolated_signal
        self.history_queue.append(self.patches.copy())
