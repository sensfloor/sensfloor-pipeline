import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloorConfig, RoIFloor
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class DatasetConfig:
    floor_config: RoIFloorConfig
    drop_landmarks: list[PoseLandmark] | None = None
    rotate_data: bool = False
    normalize_signals: bool = False
    normalize_to_max: bool = False


def normalize_roi(roi: torch.Tensor, idle_floor_value: int, normalize_to_max: bool) -> torch.Tensor:
    normalizes_roi = roi - idle_floor_value
    normalize_to = normalizes_roi.max() if normalize_to_max else idle_floor_value
    return normalizes_roi / normalize_to


def rotate_pose(pose: torch.Tensor, degree: int) -> torch.Tensor:
    num_joints = pose.shape[0] // 3
    reshaped_pose = pose.reshape(num_joints, 3).T
    rad = math.radians(degree)
    # Rotation matrix for rotating around y-axis
    rotation_matrix = torch.tensor(
        [
            [math.cos(rad), 0, math.sin(rad)],
            [0, 1, 0],
            [-math.sin(rad), 0, math.cos(rad)],
        ],
    )
    rotated_pose = rotation_matrix @ reshaped_pose
    return rotated_pose.T.reshape(num_joints * 3)


def rotate_roi(roi: torch.Tensor, degree: int) -> torch.Tensor:
    if degree % 90 != 0:
        message = f"Can only rotate roi in 90° steps (tried rotating by {degree}°)..."
        raise ValueError(message)

    num_90_rotations = degree // 90
    return roi.rot90(k=num_90_rotations, dims=(1, 2))


def drop_landmarks(poses: pd.DataFrame, drop_landmarks: list[PoseLandmark]) -> pd.DataFrame:
    columns_to_drop = []
    for landmark in drop_landmarks:
        columns_to_drop += [f"x{landmark.value}", f"y{landmark.value}", f"z{landmark.value}"]

    return poses.drop(columns=columns_to_drop)


def remove_noise_messages(sensfloor_readout: pd.DataFrame, noise_threshold: int) -> pd.DataFrame:
    signal_columns = ["0", "1", "2", "3", "4", "5", "6", "7"]
    non_noise_rows = sensfloor_readout[signal_columns].gt(noise_threshold).any(axis=1)
    return sensfloor_readout[non_noise_rows]


def get_unique_frames_with_poses(sensfloor_readout: pd.DataFrame, poses: pd.DataFrame) -> np.ndarray:
    # Get all the frames with messages
    unique_readout_frames = sensfloor_readout["frame_number"].unique()
    # Check if for these frames also poses exist
    frames_containing_messages_mask = np.isin(unique_readout_frames, poses["frame"].unique())
    # Return all frame unique numbers for which poses exist
    return unique_readout_frames[frames_containing_messages_mask]


# TODO: go through all unique frames with pose and save signals in dataframe and filter in dataset for the id -> get all values for that sequence
def get_sequences(sensfloor_readout: pd.DataFrame, poses: pd.DataFrame) -> list[tuple[int,int]]:
    # Get all the frames with messages
    unique_readout_frames = sensfloor_readout["frame_number"].unique()
    # Check if for these frames also poses exist
    frames_containing_messages_mask = np.isin(unique_readout_frames, poses["frame"].unique())
    # Return all frame unique numbers for which poses exist
    unique_frames_with_poses = unique_readout_frames[frames_containing_messages_mask]

    sequence_counts_with_frame = []
    last_frame = unique_frames_with_poses[0]
    frames_in_sequence_count = 0
    first_frame_in_sequence = last_frame
    for frame in unique_frames_with_poses:
        frame_diff = frame - last_frame
        if frame_diff < 15:
            frames_in_sequence_count += 1
        else:
            sequence_counts_with_frame.append((int(frame - first_frame_in_sequence), int(last_frame)))
            first_frame_in_sequence = frame

        last_frame = frame

    return sequence_counts_with_frame


@dataclass
class DetailedSensfloorPosesData:
    frame_number: int
    floor: RoIFloor
    untransformed_roi_tensor: torch.Tensor
    transformed_roi_tensor: torch.Tensor
    untransformed_label_tensor: torch.Tensor
    transformed_label_tensor: torch.Tensor


