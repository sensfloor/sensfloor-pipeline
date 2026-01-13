import argparse
from pathlib import Path

import mediapipe as mp

from data_collection.mediapipe_pose_extraction import read_video


def main(date: str | None) -> None:
    model_path = "./data_collection/pose_landmarker_full.task"

    BaseOptions = mp.tasks.BaseOptions  # TODO: Check if just an import (and use it as one)
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # options for Video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )

    DATA_PATH = Path("data")
    if date is not None:
        read_video(
            DATA_PATH / date / "video.mp4",
            options,
            draw_image=False,
            generate_new_header=False,
        )
    else:
        print("No video specified. Extracting all videos")
        for directory in DATA_PATH.iterdir():
            if directory.is_dir():
                print(f"Extracting {directory}")
                read_video(
                    DATA_PATH / directory.name / "video.mp4",
                    options,
                    draw_image=False,
                    generate_new_header=False,
                )


if __name__ == "__main__":
    # 1. Create the parser
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Pose Landmarker on a specified video file.",
    )

    # 2. Add the video path argument
    parser.add_argument(
        "--date",
        type=str,
        help="The date of the video to be extracted. Extracting all from data otherwise (e.g., 2025-12-01_12-44-43)",
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Call the main function with the path from the arguments
    # We wrap the argument in Path() to ensure it's a pathlib.Path object
    main(args.date)
