from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloor



@dataclass
class DatasetConfig:
    history_maxlen: int
    floor_size_x: int
    floor_size_y: int
    roi_size: int
    drop_landmarks: list[PoseLandmark] | None = None
    signal_threshold: int = 140

class SensfloorPosesDataset(Dataset):
    def __init__(
        self,
        poses_df: pd.DataFrame,
        sensfloor_readout_df: pd.DataFrame,
        config: DatasetConfig
    ) -> None:
        super().__init__()
        self.poses_df = poses_df
        self.sensfloor_readout_df = sensfloor_readout_df
        self.history_maxlen = config.history_maxlen
        self.floor_size_x = config.floor_size_x
        self.floor_size_y = config.floor_size_y
        self.roi_size = config.roi_size

        if config.drop_landmarks:
            columns_to_drop = []
            for landmark in config.drop_landmarks:
                columns_to_drop += [f"x{landmark}", f"y{landmark}", f"z{landmark}"]

            self.poses_df = self.poses_df.drop(columns=columns_to_drop)

        # Remove all messages that are below a signal value of 140 (no activity, just noise)
        relevant_rows = self.sensfloor_readout_df[["0", "1", "2", "3", "4", "5", "6", "7"]].gt(config.signal_threshold).any(axis=1)
        filtered_readout = self.sensfloor_readout_df[relevant_rows]
        frames_containing_messages = filtered_readout["frame_number"].unique()
        frames_containing_messages_mask = np.isin(frames_containing_messages, self.poses_df["frame"].unique())
        self.frames_containing_messages = frames_containing_messages[frames_containing_messages_mask]

    def __len__(self) -> int:
        return len(self.frames_containing_messages)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get signal history
        frame_number = self.frames_containing_messages[index]
        floor = self._create_roi_floor(frame_number)
        roi = floor.get_roi()

        if roi is None:
            error_message = "No region of interest found..."
            raise RuntimeError(error_message)

        roi_tensor = torch.Tensor(roi.history)
        normalized_roi_tensor = self._normalize_roi(roi_tensor)

        # Get pose
        label = self.poses_df[self.poses_df["frame"] == frame_number].drop(columns=["frame"]).to_numpy()[0]
        label_tensor = torch.Tensor(label)

        return normalized_roi_tensor, label_tensor

    def _normalize_roi(self, roi: torch.Tensor) -> torch.Tensor:
        # TODO: Maybe normalize by maximum value of roi to mitigate the effect of different footwear
        normalized_roi = (roi - 127) / 127
        return normalized_roi

    def _create_roi_floor(self, frame_number: int) -> RoIFloor:
        floor = RoIFloor(
            self.floor_size_x,
            self.floor_size_y,
            self.history_maxlen,
            self.roi_size,
        )
        earliest_frame_included = frame_number - self.history_maxlen
        for current_frame_number in range(earliest_frame_included, frame_number + 1):
            messages = self.sensfloor_readout_df[self.sensfloor_readout_df["frame_number"] == current_frame_number]
            positions = (
                messages[["x", "y"]].to_numpy() - 1
            ).tolist()  # Subtract 1 from positions because they start at 1,1
            signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy()
            floor.update(positions, signals)
        return floor
