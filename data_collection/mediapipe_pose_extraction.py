import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker
from tqdm import tqdm

from mediapipe_utils import get_landmarks_header

# use "generate_new_header" flag and copy printout
HEADER = ['frame', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'x5', 'y5', 'z5', 'x6', 'y6', 'z6', 'x7', 'y7', 'z7', 'x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10', 'z10', 'x11', 'y11', 'z11', 'x12', 'y12', 'z12', 'x13', 'y13', 'z13', 'x14', 'y14', 'z14', 'x15', 'y15', 'z15', 'x16', 'y16', 'z16', 'x17', 'y17', 'z17', 'x18', 'y18', 'z18', 'x19', 'y19', 'z19', 'x20', 'y20', 'z20', 'x21', 'y21', 'z21', 'x22', 'y22', 'z22', 'x23', 'y23', 'z23', 'x24', 'y24', 'z24', 'x25', 'y25', 'z25', 'x26', 'y26', 'z26', 'x27', 'y27', 'z27', 'x28', 'y28', 'z28', 'x29', 'y29', 'z29', 'x30', 'y30', 'z30', 'x31', 'y31', 'z31', 'x32', 'y32', 'z32']


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


def write_csv_header(filepath: str, header: list[str]):
    """Overwrite filepath and create new csv file with header"""
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)


def landmarks_to_row(pose_landmarker_result, frame) -> list | None:
    """Convert pose landmarks to a list matching a CSV row."""
    if not pose_landmarker_result.pose_world_landmarks:
        return None
    landmarks = pose_landmarker_result.pose_world_landmarks[0]
    row = [frame]
    for l in landmarks:
        row.extend([l.x, l.y, l.z])
    return row


def read_video(p: Path, options: PoseLandmarkerOptions, draw_image: bool, generate_new_header: bool):
    """
    read all frames of a video and write the mediapipe 3D poses into a csv file
    video_path: path to video file
    options: Options for Mediapipe
    draw_image: Show each frame with the landmark predictions
    generate_new_header: use the first frame to generate a csv header, if set to False, uses HEADER constant instead
    """
    csv_path = f"{p.parent / p.stem}_poses.csv"
    if os.path.exists(csv_path):
        print(f"File {csv_path} already exists, skipping")
        return
    write_csv_header(csv_path, HEADER)

    with (PoseLandmarker.create_from_options(options) as landmarker, \
            open(csv_path, "a", newline='') as f):

        # --- Initializing ---
        print("Pose Landmarker initialized.")
        writer = csv.writer(f)

        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise RuntimeError(f"Error: Could not open {p}")

        # --- Progress bar ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        p_bar = tqdm(range(total_frames), desc="Processing frames", unit="frame")

        while True:
            # --- Processing Image ---
            frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, numpy_frame_from_opencv = cap.read()

            if not ret:
                print("Failed to capture frame. Exiting")
                break

            rgb_image = cv2.cvtColor(numpy_frame_from_opencv, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            frame_timestamp_ms = int(time.time() * 1000)

            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # --- Writing to csv ---
            row = landmarks_to_row(pose_landmarker_result, frame)
            if row:
                if frame == 0:
                    header = HEADER
                    if generate_new_header:
                        header = get_landmarks_header(pose_landmarker_result)
                        print(f"HEADER = {header}")
                    write_csv_header(csv_path, header)
                writer.writerow(row)

            # --- Visualization ---
            if draw_image:
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
                cv2.imshow("MediaPipe Pose Landmarker", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break

            p_bar.update()

        cap.release()
        cv2.destroyAllWindows()


def main(date: str):
    model_path = './data_collection/pose_landmarker_full.task'

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # options for Video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)

    DATA_PATH = Path("data")
    if date is not None:
        read_video(DATA_PATH / date / "video.mp4", options, draw_image=False, generate_new_header=False)
    else:
        print("No video specified. Extracting all videos")
        for date in DATA_PATH.iterdir():
            print(f"Extracting {date}")
            read_video(DATA_PATH / date.name / "video.mp4", options, draw_image=False, generate_new_header=False)

if __name__ == '__main__':
    # 1. Create the parser
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Pose Landmarker on a specified video file."
    )

    # 2. Add the video path argument
    parser.add_argument(
        '--date',
        type=str,
        help="The date of the video to be extracted. Extracting all from data otherwise (e.g., 2025-12-01_12-44-43)"
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Call the main function with the path from the arguments
    # We wrap the argument in Path() to ensure it's a pathlib.Path object
    main(args.date)