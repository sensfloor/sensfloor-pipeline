import argparse
from pathlib import Path

import mediapipe as mp

from data_collection.mediapipe_pose_extraction import read_video


def main(date: str | None, data_path: Path, model_path: Path) -> None:
    # options for Video mode:
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )

    if date is not None:
        read_video(
            data_path / date / "video.mp4",
            options,
            draw_image=False,
            generate_new_header=False,
        )
    else:
        print("No video specified. Extracting all videos")
        for directory in data_path.iterdir():
            if directory.is_dir():
                print(f"Extracting {directory}")
                read_video(
                    data_path / directory.name / "video.mp4",
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
        default=Path("data"),
        help="Root folder of recorded data.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data_collection/pose_landmarker_full.task"),
        help="Path to pose estimation model.",
    )

    parser.add_argument(
        "--date",
        type=str,
        help="The date of the video to be extracted. Extracting all from data otherwise (e.g., 2025-12-01_12-44-43)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    main(date=args.date, data_path=args.data, model_path=args.model)
