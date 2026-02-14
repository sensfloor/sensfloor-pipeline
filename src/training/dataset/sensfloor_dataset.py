import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_loading.roi_floor import RoIFloor, create_roi_floor
from src.training.dataset.dataset_utils import (
    DatasetConfig,
    DetailedSensfloorPosesData,
    drop_unused_landmarks,
    get_unique_frames_with_poses,
    remove_noise_messages,
)
from src.training.dataset.transformations import normalize_roi, rotate_pose, rotate_roi


class SensfloorPosesDataset(Dataset):
    def __init__(
        self,
        poses_df: pd.DataFrame,
        sensfloor_readout_df: pd.DataFrame,
        config: DatasetConfig,
        return_detailed: bool = False,
    ) -> None:
        super().__init__()
        self.poses_df = poses_df
        self.sensfloor_readout_df = sensfloor_readout_df
        self.config = config
        self.return_detailed = return_detailed

        if config.landmarks:
            self.poses_df = drop_unused_landmarks(poses=poses_df, landmarks_to_keep=config.landmarks)

        # Remove all messages that are below a specified signal value (no activity, just noise)
        self.filtered_readout = remove_noise_messages(
            sensfloor_readout=self.sensfloor_readout_df,
            noise_threshold=config.floor_config.active_field_min_value,
        )

        self.frames_containing_messages = get_unique_frames_with_poses(
            sensfloor_readout=self.filtered_readout,
            poses=poses_df,
        )

    def __len__(self) -> int:
        return len(self.frames_containing_messages)

    def __getitem__(self, index: int) -> DetailedSensfloorPosesData | tuple[torch.Tensor, torch.Tensor]:
        data = self.get_detailed_data(index)
        if self.return_detailed:
            return data
        return data.transformed_roi_tensor, data.transformed_label_tensor

    def get_detailed_data(self, index: int) -> DetailedSensfloorPosesData:
        # Get signal history
        frame_number = self.frames_containing_messages[index]
        floor = create_roi_floor(self.config.floor_config, self.sensfloor_readout_df, frame_number)
        return self._get_transformed_data(floor, frame_number)

    def _get_transformed_data(self, floor: RoIFloor, frame_number: int) -> DetailedSensfloorPosesData:
        roi = floor.get_roi()
        if roi is None:
            error_message = "No region of interest found..."
            raise RuntimeError(error_message)

        untransformed_roi_tensor = torch.Tensor(roi.history)

        transformed_roi_tensor = untransformed_roi_tensor
        if self.config.normalize_signals:
            transformed_roi_tensor = normalize_roi(
                untransformed_roi_tensor,
                self.config.floor_config.idle_field_value,
                self.config.normalize_to_max,
            )

        # Get pose
        label = self.poses_df[self.poses_df["frame"] == frame_number].drop(columns=["frame"]).to_numpy()[0]
        untransformed_label_tensor = torch.Tensor(label)

        transformed_label_tensor = untransformed_label_tensor
        if self.config.rotate_data:
            degree = random.choice([0, 90, 180, 270])
            transformed_label_tensor = rotate_pose(untransformed_label_tensor, degree)
            transformed_roi_tensor = rotate_roi(transformed_roi_tensor, degree)

        return DetailedSensfloorPosesData(
            frame_number=frame_number,
            floor=floor,
            untransformed_roi_tensor=untransformed_roi_tensor,
            transformed_roi_tensor=transformed_roi_tensor,
            untransformed_label_tensor=untransformed_label_tensor,
            transformed_label_tensor=transformed_label_tensor,
        )
