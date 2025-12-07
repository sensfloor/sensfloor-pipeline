from collections import deque

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


class Floor:
    def __init__(self, x_size: int, y_size: int, history_maxlen: int) -> None:
        self._x_size = x_size
        self._y_size = y_size
        self._history_maxlen = history_maxlen
        self._history_queue = deque(maxlen=history_maxlen)
        self.patches = np.ones((x_size * 4, y_size * 4)) * 127

    @property
    def shape(self) -> tuple[int, int]:
        return self._x_size, self._y_size

    @property
    def history(self) -> np.ndarray:
        empty_history = self._history_maxlen - len(self._history_queue)
        zero_fill = np.ones((empty_history, self._x_size * 4, self._y_size * 4)) * 127
        return np.stack([*zero_fill, *list(self._history_queue)])

    def update(self, positions: np.ndarray, signals: np.ndarray) -> None:
        clipped_signals = signals.clip(min=127)
        for (x, y), signal in zip(positions, clipped_signals, strict=True):
            interpolated_signal = interpolate_signal(signal)
            x_patches = x * 4
            y_patches = y * 4
            self.patches[x_patches : x_patches + 4, y_patches : y_patches + 4] = interpolated_signal
        self._history_queue.append(self.patches.copy())
