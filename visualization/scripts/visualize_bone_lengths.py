import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loading.pose_landmark import PoseLandmark


@dataclass(frozen=True)
class Bone:
    name: str
    landmark_a: PoseLandmark
    landmark_b: PoseLandmark


BONES_TO_VISUALIZE = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize bone length deviations.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Required path to input CSV with collected pose labels.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the plot. If omitted, the plot is shown.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.is_file():
        message = f"CSV not found: {args.csv}"
        raise RuntimeError(message)

    poses_df = pd.read_csv(args.csv)

    for bone in BONES_TO_VISUALIZE:
        a, b = bone.landmark_a.value, bone.landmark_b.value

        pa = poses_df[[f"x{a}", f"y{a}", f"z{a}"]].to_numpy()
        pb = poses_df[[f"x{b}", f"y{b}", f"z{b}"]].to_numpy()

        poses_df[f"{bone.name}_length"] = np.linalg.norm(pa - pb, axis=1)

    length_cols = [f"{bone.name}_length" for bone in BONES_TO_VISUALIZE]

    length_df = poses_df[length_cols]
    relative_deviations_df = (length_df - length_df.median()) / length_df.median()

    plt.figure(figsize=(10, 5))

    plt.boxplot(
        [relative_deviations_df[columns].to_numpy() for columns in relative_deviations_df.columns],
        tick_labels=length_cols,
        showfliers=False,
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.ylabel("Relative deviation from median")
    plt.title("Relative bone length deviation")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out)


if __name__ == "__main__":
    main()
