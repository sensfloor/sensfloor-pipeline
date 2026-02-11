import argparse
from pathlib import Path

import mediapipe as mp

from src.data_collection.mediapipe_pose_extraction import read_video
from src.definitions import TRAIN_DATA_PATH, ROOT_PATH


def main(data_path: Path, model_path: Path) -> None:
    # options for Video mode:
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )

    print(f"Extracting all videos in folder {data_path}")
    for directory in data_path.iterdir():
        if directory.is_dir():
            print(f"Extracting {directory}")
            read_video(
                directory,
                options,
                draw_image=False,
                generate_new_header=False,
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Pose Landmarker on a specified video file.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=TRAIN_DATA_PATH,
        help="Root folder of recorded data.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT_PATH / "src" / "data_collection" / "pose_landmarker_full.task",
        help="Path to pose estimation model.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    main(data_path=args.data, model_path=args.model)
