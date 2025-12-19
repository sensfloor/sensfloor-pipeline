import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage


def draw_floor(ax: Axes, floor: np.ndarray, cmap: Colormap, vmin: int = 127, vmax: int = 255) -> AxesImage:
    x_size_floor_array, y_size_floor_array = floor.shape

    image = ax.imshow(
        floor,
        origin="lower",
        cmap=cmap,
        extent=(0, y_size_floor_array, 0, x_size_floor_array),
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    ax.grid(which="major", color="w", linestyle="-", linewidth=0.5, alpha=0.3)

    fields_per_patch = 4
    x_positions = np.arange(0, y_size_floor_array + 1, fields_per_patch)
    y_positions = np.arange(0, x_size_floor_array + 1, fields_per_patch)

    x_labels = [str(int(x_position / fields_per_patch)) for x_position in x_positions]
    y_labels = [str(int(y_position / fields_per_patch)) for y_position in y_positions]

    ax.set_xticks(x_positions, x_labels)
    ax.set_yticks(y_positions, y_labels)

    ax.set_xlabel("y")
    ax.set_ylabel("x")

    return image


def draw_roi(ax: Axes, x: int, y: int, roi_size: int) -> patches.Patch:
    roi_row_start = x * 4
    roi_col_start = y * 4

    roi_rect = patches.Rectangle(
        (roi_col_start, roi_row_start),
        roi_size * 4,
        roi_size * 4,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
        linestyle="-",
        zorder=10,
    )
    return ax.add_patch(roi_rect)
