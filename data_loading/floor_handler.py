from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.ticker import FuncFormatter


class FloorHandler:
    def __init__(self, x_size: int, y_size: int, history_len: int, kernel_size: int) -> None:
        self.history = deque(maxlen=history_len)
        self.history.append(np.zeros((x_size * 4, y_size * 4)))
        self.roi_center = np.ndarray([])

        if kernel_size % 2 == 0:
            error_message = f"Kernel size must be an odd number (given: {kernel_size})"
            raise ValueError(error_message)

        self.kernel_size = kernel_size
        self.kernel_offset = kernel_size // 2

    def update_floor(self, signals: np.ndarray, positions: np.ndarray) -> None:
        new_floor = self.history[-1].copy()
        self.roi_center = np.unique(positions, axis=0)
        for signal, position in zip(signals, positions, strict=True):
            signal[signal < 140] = 0
            interpolated_signal = self.interpolate_signal(signal)
            x_floor = position[0] * 4
            y_floor = position[1] * 4
            new_floor[x_floor : x_floor + 4, y_floor : y_floor + 4] = interpolated_signal
        self.history.append(new_floor)

    def interpolate_signal(self, signal: np.ndarray) -> np.ndarray:
        averages = signal.reshape((4, 2)).mean(axis=1)
        return np.array(
            [
                [averages[3], signal[7], signal[0], averages[0]],
                [signal[6], averages[3], averages[0], signal[1]],
                [signal[5], averages[2], averages[1], signal[2]],
                [averages[2], signal[4], signal[3], averages[1]],
            ],
        )

    def get_roi(self) -> tuple[int, int, np.ndarray]:
        last_floor = self.history[-1]

        roi = np.zeros((self.kernel_size * 4, self.kernel_size * 4))
        max_score = 0
        x_roi = 0
        y_roi = 0
        for x, y in self.roi_center:
            x_corner_tl = (
                np.min(
                    [
                        np.max([0, x - self.kernel_offset]),
                        (int(last_floor.shape[0] / 4) - 1) - (self.kernel_offset * 2),
                    ],
                )
                * 4
            )
            x_corner_br = x_corner_tl + self.kernel_size * 4
            y_corner_tl = (
                np.min(
                    [
                        np.max([0, y - self.kernel_offset]),
                        (int(last_floor.shape[1] / 4) - 1) - (self.kernel_offset * 2),
                    ],
                )
                * 4
            )
            y_corner_br = y_corner_tl + self.kernel_size * 4
            roi_candidate = last_floor[x_corner_tl:x_corner_br, y_corner_tl:y_corner_br]
            score = roi_candidate.sum()

            if score > max_score:
                max_score = score
                roi = roi_candidate
                x_roi = x_corner_tl / 4
                y_roi = y_corner_tl / 4

        return x_roi, y_roi, roi

    def draw(self) -> None:
        floor = self.history[-1]

        h, w = floor.shape
        plt.imshow(floor, cmap="viridis", interpolation="nearest", extent=(0, w, h, 0))
        plt.colorbar()

        ax = plt.gca()

        ax.set_xticks(np.arange(0, w + 1))
        ax.set_yticks(np.arange(0, h + 1))
        ax.grid(color="white", linewidth=0.5)

        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: int(v / 4)))  # pyright: ignore[reportArgumentType] # noqa: ARG005
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: int(v / 4)))  # pyright: ignore[reportArgumentType] # noqa: ARG005

        x, y, _ = self.get_roi()
        x0 = x * 4
        y0 = y * 4
        x1 = (x + self.kernel_size) * 4
        y1 = (y + self.kernel_size) * 4

        width = x1 - x0
        height = y1 - y0

        rect = patches.Rectangle(
            (y0, x0),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        plt.title(f"ROI at {int(x)}, {int(y)}")
        plt.show()
