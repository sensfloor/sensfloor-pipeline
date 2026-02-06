import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loading.roi_floor import RoIFloorConfig
from src.definitions import ROOT_PATH
from src.training.configs import DatasetType
from src.training.dataset.load_data import load_single_dataset
from src.training.dataset.sensfloor_dataset import DatasetConfig
from src.visualization.utils import draw_floor

DATA_PATH = ROOT_PATH / "data/hold_out/2025-12-16_12-07-42-rikuto-shorts"
DATA_INDEX = 120
df = pd.read_csv(DATA_PATH / "video_poses.csv")

floor_config = RoIFloorConfig(x_size=6, y_size=4, history_maxlen=10, roi_size=3)

dataset_config = DatasetConfig(
    floor_config=floor_config,
    normalize_signals=False,
    rotate_data=True,
)

dataset = load_single_dataset(
    data_path=DATA_PATH,
    config=dataset_config,
    dataset_type=DatasetType.HISTORY,
)  # TODO dataset_type

FRAME_TO_SHOW = 906
CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]


def visualize_rotation() -> None:
    data = dataset.get_detailed_data(DATA_INDEX)

    # Rotate mediapipe skeleton landmarks
    rotated_t = data.transformed_label_tensor.reshape(33, 3).T

    plot_xs = rotated_t[0, :].tolist()  # rotated_xs  # mediapipe X is horizontal axis
    plot_ys = rotated_t[2, :].tolist()  # rotated_zs  # mediapipe Z is vertical axis
    plot_zs = [-y for y in rotated_t[1, :].tolist()]  # mediapipe Y is depth (and inverted)

    floor_tensor = data.transformed_roi_tensor[-1].numpy()

    fig, axes = plt.subplot_mosaic(
        [["Pose", "Floor"]],
        per_subplot_kw={
            "Pose": {"projection": "3d"},
            "Floor": {},
        },
        figsize=(15, 7),
    )

    # Plot landmarks
    ax_pose = axes["Pose"]
    ax_pose.scatter(plot_xs, plot_ys, plot_zs, c="r", marker="o")

    # Draw bones
    for start, end in CONNECTIONS:
        ax_pose.plot(
            [plot_xs[start], plot_xs[end]],
            [plot_ys[start], plot_ys[end]],
            [plot_zs[start], plot_zs[end]],
            c="blue",
        )

    ax_pose.set_xlabel("X")
    ax_pose.set_ylabel("Y (Z of mediapipe)")
    ax_pose.set_zlabel("Z (-Y of mediapipe)")
    ax_pose.set_title("Pose")

    # Set axis limits
    all_coords = plot_xs + plot_ys + plot_zs
    max_range = (max(all_coords) - min(all_coords)) / 2.0
    mid_x = (max(plot_xs) + min(plot_xs)) * 0.5
    mid_y = (max(plot_ys) + min(plot_ys)) * 0.5
    mid_z = (max(plot_zs) + min(plot_zs)) * 0.5
    ax_pose.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_pose.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_pose.set_zlim(mid_z - max_range, mid_z + max_range)

    # Adjust point of view
    ax_pose.view_init(elev=5, azim=110)

    # --------------------
    ax_floor = axes["Floor"]
    cmap = mcolors.LinearSegmentedColormap.from_list("signal colormap", ["#FFFFFF", "#0033FF", "#FF6A00"])
    draw_floor(ax_floor, floor_tensor, cmap)

    fig.show()


visualize_rotation()
visualize_rotation()
visualize_rotation()
visualize_rotation()

plt.show()
