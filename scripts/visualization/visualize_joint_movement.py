from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.utils.data import Dataset

from src.data_loading.pose_landmark import PoseLandmark
from src.data_loading.roi_floor import RoIFloorConfig
from src.definitions import TRAIN_DATA_PATH
from src.training.configs import DatasetType
from src.training.dataset.dataset_utils import DatasetConfig
from src.training.dataset.load_data import load_all_datasets

JOINTS_TO_COMPARE = [
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
]

LANDMARK_GROUPS = {
    "Body": [
        PoseLandmark.LEFT_SHOULDER.name,
        PoseLandmark.RIGHT_SHOULDER.name,
        PoseLandmark.LEFT_HIP.name,
        PoseLandmark.RIGHT_HIP.name,
        PoseLandmark.NOSE.name,
        PoseLandmark.LEFT_EYE_INNER.name,
        PoseLandmark.LEFT_EYE.name,
        PoseLandmark.LEFT_EYE_OUTER.name,
        PoseLandmark.RIGHT_EYE_INNER.name,
        PoseLandmark.RIGHT_EYE.name,
        PoseLandmark.RIGHT_EYE_OUTER.name,
        PoseLandmark.LEFT_EAR.name,
        PoseLandmark.RIGHT_EAR.name,
        PoseLandmark.MOUTH_LEFT.name,
        PoseLandmark.MOUTH_RIGHT.name,
    ],
    "Arms": [
        PoseLandmark.LEFT_ELBOW.name,
        PoseLandmark.RIGHT_ELBOW.name,
        PoseLandmark.LEFT_WRIST.name,
        PoseLandmark.RIGHT_WRIST.name,
        PoseLandmark.LEFT_PINKY.name,
        PoseLandmark.RIGHT_PINKY.name,
        PoseLandmark.LEFT_INDEX.name,
        PoseLandmark.RIGHT_INDEX.name,
        PoseLandmark.LEFT_THUMB.name,
        PoseLandmark.RIGHT_THUMB.name,
    ],
    "Legs": [
        PoseLandmark.LEFT_KNEE.name,
        PoseLandmark.RIGHT_KNEE.name,
        PoseLandmark.LEFT_ANKLE.name,
        PoseLandmark.RIGHT_ANKLE.name,
        PoseLandmark.LEFT_HEEL.name,
        PoseLandmark.RIGHT_HEEL.name,
        PoseLandmark.LEFT_FOOT_INDEX.name,
        PoseLandmark.RIGHT_FOOT_INDEX.name,
    ],
}

CATEGORY_COLORS = {
    "Body": "#95a5a6",
    "Arms": "#f45f74",
    "Legs": "#00b0be",
}


def get_landmark_category(name: str) -> str:
    upper_name = name.upper()
    for category, landmarks in LANDMARK_GROUPS.items():
        for landmark in landmarks:
            if landmark in upper_name:
                return category
    return "Body"


def clean_label(landmark: PoseLandmark) -> str:
    clean = landmark.name.lower().replace("_", " ").title()
    return clean.replace("Left", "L.").replace("Right", "R.")


def pair_base(label: str) -> str:
    return label.replace("L.", "").replace("R.", "").strip()


def plot_joint_distance_boxplot(
    distances: torch.Tensor | np.ndarray,
) -> None:
    distances = distances.detach().cpu().numpy() if isinstance(distances, torch.Tensor) else np.asarray(distances)

    _, num_joints = distances.shape

    distances_in_cm = distances * 100.0

    plt.rcParams.update(
        {
            "font.family": "Courier New",
            "font.size": 16,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.color": "black",
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
        },
    )

    labels = [clean_label(lm) for lm in JOINTS_TO_COMPARE]
    plot_data = [distances_in_cm[:, j][~np.isnan(distances_in_cm[:, j])] for j in range(num_joints)]
    box_colors = [CATEGORY_COLORS[get_landmark_category(lm.name)] for lm in JOINTS_TO_COMPARE]

    positions = []
    current_pos = 1.0
    for i in range(num_joints):
        positions.append(current_pos)
        if i < num_joints - 1:
            if pair_base(labels[i]) == pair_base(labels[i + 1]):
                current_pos += 0.5
            else:
                current_pos += 0.8

    fig, ax = plt.subplots(figsize=(5, 3))

    bplot = ax.boxplot(
        plot_data,
        positions=positions,
        patch_artist=True,
        tick_labels=labels,
        widths=0.4,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4, "markeredgecolor": "black"},
    )

    for patch, color in zip(bplot["boxes"], box_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_linewidth(0.75)
        patch.set_alpha(1)

    plt.setp(bplot["medians"], color="black", linewidth=1.0)

    ax.set_ylabel("Distance to mean pose (cm)")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha="center")

    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["Body"], edgecolor="black", label="Body"),
        Patch(facecolor=CATEGORY_COLORS["Arms"], edgecolor="black", label="Arms"),
        Patch(facecolor=CATEGORY_COLORS["Legs"], edgecolor="black", label="Legs"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout(pad=0.0)
    plt.margins(0, 0)

    output_path = Path("outputs/images/joint_movement.svg")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0)
    print(f"Saved boxplot to {output_path}")
    plt.close(fig)


def sample_poses(dataset: Dataset, n_samples: int) -> torch.Tensor:
    n = min(n_samples, len(dataset))  # type: ignore
    idxs = torch.randperm(len(dataset))[:n]  # type: ignore
    joint_idxs = torch.tensor([joint.value for joint in JOINTS_TO_COMPARE], dtype=torch.long)

    poses = []
    for i in idxs.tolist():
        _, pose = dataset[i]
        pose = pose.reshape(-1, 3)
        pose = pose.index_select(0, joint_idxs)
        poses.append(torch.as_tensor(pose))
    return torch.stack(poses, dim=0)


def distances_to_mean(poses: torch.Tensor) -> torch.Tensor:
    ref_pose = poses.mean(dim=0)
    return torch.linalg.norm(poses - ref_pose.unsqueeze(0), dim=2)


def main() -> None:
    floor_config = RoIFloorConfig(
        x_size=6,
        y_size=4,
        history_maxlen=15,
        roi_size=4,
        active_field_min_value=140,
    )

    dataset_config = DatasetConfig(
        floor_config=floor_config,
        normalize_signals=False,
    )

    _, _, test_dataset = load_all_datasets(
        data_root_path=TRAIN_DATA_PATH,
        config=dataset_config,
        ratios=(0.8, 0.1, 0.1),
        dataset_type=DatasetType.HISTORY,
    )

    sampled_poses = sample_poses(test_dataset, 10000)
    distances = distances_to_mean(sampled_poses)

    plot_joint_distance_boxplot(distances)


if __name__ == "__main__":
    main()
