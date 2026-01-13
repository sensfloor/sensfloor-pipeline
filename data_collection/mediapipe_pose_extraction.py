import csv
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker
from tqdm import tqdm

from data_collection.mediapipe_utils import HEADER, get_landmarks_header


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                )
                for landmark in pose_landmarks
            ],
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def write_csv_header(filepath: str, header: list[str]):
    """Overwrite filepath and create new csv file with header"""
    with open(filepath, "w", newline="") as f:
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


def read_video(
    p: Path,
    options: PoseLandmarkerOptions,
    draw_image: bool,
    generate_new_header: bool,
):
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

    with (
        PoseLandmarker.create_from_options(options) as landmarker,
        open(csv_path, "a", newline="") as f,
    ):
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

            pose_landmarker_result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms,
            )

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
                annotated_image = draw_landmarks_on_image(
                    mp_image.numpy_view(),
                    pose_landmarker_result,
                )
                cv2.imshow(
                    "MediaPipe Pose Landmarker",
                    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
                )

                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break

            p_bar.update()

        cap.release()
        cv2.destroyAllWindows()
