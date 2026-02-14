from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data_loading.pose_landmark import PoseLandmark
from src.data_loading.roi_floor import RoIFloor, RoIFloorConfig


@dataclass(frozen=True)
class DatasetConfig:
    floor_config: RoIFloorConfig
    landmarks: list[PoseLandmark] | None = None
    rotate_data: bool = False
    normalize_signals: bool = False
    normalize_to_max: bool = False


def drop_unused_landmarks(poses: pd.DataFrame, landmarks_to_keep: list[PoseLandmark]) -> pd.DataFrame:
    drop_columns = []
    for landmark in PoseLandmark:
        if landmark not in landmarks_to_keep:
            drop_columns += [f"x{landmark.value}", f"y{landmark.value}", f"z{landmark.value}"]
    return poses.drop(columns=drop_columns)


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


def get_sequences(sensfloor_readout: pd.DataFrame, poses: pd.DataFrame) -> list[tuple[int, int]]:
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
