from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loading.pose_landmark import PoseLandmark

POSES_PATH = Path("data/2025-12-16_20-45-38-line-felix/video_poses.csv")
poses_df = pd.read_csv(POSES_PATH)


@dataclass(frozen=True)
class Bone:
    name: str
    landmark_a: PoseLandmark
    landmark_b: PoseLandmark


bones = [
    Bone("sole_left", PoseLandmark.LEFT_FOOT_INDEX, PoseLandmark.LEFT_HEEL),
    Bone("sole_right", PoseLandmark.RIGHT_FOOT_INDEX, PoseLandmark.RIGHT_HEEL),
    Bone("lower_leg_left", PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_KNEE),
    Bone("lower_leg_right", PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_KNEE),
    Bone("thigh_left", PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    Bone("thigh_right", PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    Bone("pelvis", PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_HIP),
    Bone("shoulders", PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),
    Bone("torso_left", PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER),
    Bone("torso_right", PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_SHOULDER),
]

for bone in bones:
    a, b = bone.landmark_a.value, bone.landmark_b.value

    pa = poses_df[[f"x{a}", f"y{a}", f"z{a}"]].to_numpy()
    pb = poses_df[[f"x{b}", f"y{b}", f"z{b}"]].to_numpy()

    poses_df[f"{bone.name}_length"] = np.linalg.norm(pa - pb, axis=1)


length_cols = [f"{bone.name}_length" for bone in bones]

length_df = poses_df[length_cols]
relative_deviations_df = (length_df - length_df.median()) / length_df.median()

plt.figure(figsize=(10, 5))

plt.boxplot(
    [relative_deviations_df[columns].to_numpy() for columns in relative_deviations_df.columns],
    labels=length_cols,  # type: ignore  # noqa: PGH003
    showfliers=False,
)

plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Relative deviation from median")
plt.title("Relative bone length deviation")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
