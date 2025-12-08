from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloorConfig, create_roi_floor


@dataclass(frozen=True)
class DatasetConfig:
    floor_config: RoIFloorConfig
    drop_landmarks: list[PoseLandmark] | None = None
    signal_threshold: int = 140
    normalize_signals: bool = False


def normalize_roi(roi: torch.Tensor) -> torch.Tensor:
    # TODO: Maybe normalize by maximum value of roi to mitigate the effect of different footwear
    return (roi - 127) / 127


def drop_landmarks(poses: pd.DataFrame, drop_landmarks: list[PoseLandmark]) -> pd.DataFrame:
    columns_to_drop = []
    for landmark in drop_landmarks:
        columns_to_drop += [f"x{landmark}", f"y{landmark}", f"z{landmark}"]

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


def clip_field_values(
    sensfloor_readout: pd.DataFrame,
    signal_threshold: int,
    empty_field_signal_value: int,
) -> pd.DataFrame:
    signal_columns = ["0", "1", "2", "3", "4", "5", "6", "7"]
    empty_field_signal_value = 127
    sensfloor_readout[signal_columns] = sensfloor_readout[signal_columns].mask(
        sensfloor_readout[signal_columns] < signal_threshold,
        empty_field_signal_value,
    )
    return sensfloor_readout


class SensfloorPosesDataset(Dataset):
    def __init__(
        self,
        poses_df: pd.DataFrame,
        sensfloor_readout_df: pd.DataFrame,
        config: DatasetConfig,
    ) -> None:
        super().__init__()
        self.poses_df = poses_df
        self.sensfloor_readout_df = sensfloor_readout_df
        self.config = config

        self.sensfloor_readout_df = clip_field_values(
            self.sensfloor_readout_df,
            signal_threshold=config.signal_threshold,
            empty_field_signal_value=127,
        )

        if config.drop_landmarks:
            self.poses_df = drop_landmarks(poses=poses_df, drop_landmarks=config.drop_landmarks)

        # Remove all messages that are below a signal value of 140 (no activity, just noise)
        filtered_readout = remove_noise_messages(
            sensfloor_readout=self.sensfloor_readout_df,
            noise_threshold=config.signal_threshold,
        )

        self.frames_containing_messages = get_unique_frames_with_poses(
            sensfloor_readout=filtered_readout,
            poses=poses_df,
        )

    def __len__(self) -> int:
        return len(self.frames_containing_messages)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get signal history
        frame_number = self.frames_containing_messages[index]
        floor = create_roi_floor(self.config.floor_config, self.sensfloor_readout_df, frame_number)
        roi = floor.get_roi()

        if roi is None:
            error_message = "No region of interest found..."
            raise RuntimeError(error_message)

        roi_tensor = torch.Tensor(roi.history)

        if self.config.normalize_signals:
            roi_tensor = normalize_roi(roi_tensor)

        # Get pose
        label = self.poses_df[self.poses_df["frame"] == frame_number].drop(columns=["frame"]).to_numpy()[0]
        label_tensor = torch.Tensor(label)

        return roi_tensor, label_tensor
