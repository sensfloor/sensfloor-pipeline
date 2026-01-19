import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloor, RoIFloorConfig, create_roi_floor


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


@dataclass
class DetailedSensfloorPosesData:
    frame_number: int
    floor: RoIFloor
    untransformed_roi_tensor: torch.Tensor
    transformed_roi_tensor: torch.Tensor
    untransformed_label_tensor: torch.Tensor
    transformed_label_tensor: torch.Tensor


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

        if config.drop_landmarks:
            self.poses_df = drop_landmarks(poses=poses_df, drop_landmarks=config.drop_landmarks)

        # Remove all messages that are below a specified signal value (no activity, just noise)
        filtered_readout = remove_noise_messages(
            sensfloor_readout=self.sensfloor_readout_df,
            noise_threshold=config.floor_config.active_field_min_value,
        )

        self.frames_containing_messages = get_unique_frames_with_poses(
            sensfloor_readout=filtered_readout,
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


def load_single_dataset(data_path: Path, config: DatasetConfig, return_detailed=False) -> SensfloorPosesDataset:
    poses_df = pd.read_csv(data_path / "video_poses.csv")
    readout_df = pd.read_csv(data_path / "sensfloor_readout.csv")
    return SensfloorPosesDataset(
        poses_df=poses_df,
        sensfloor_readout_df=readout_df,
        config=config,
        return_detailed = return_detailed,
    )


def load_all_datasets(data_root_path: Path, config: DatasetConfig) -> ConcatDataset[SensfloorPosesDataset]:
    folders = [folder for folder in data_root_path.iterdir() if folder.is_dir()]
    datasets = [load_single_dataset(folder, config) for folder in folders]
    return ConcatDataset(datasets)


def train_val_test_split(
        data_root_path: Path,
        ratios: tuple[float, float, float],
        config: DatasetConfig,
        batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if sum(ratios) != 1.0:
        message = "Splitting ratios don't add up to 1!"
        raise RuntimeError(message)

    dataset = load_all_datasets(data_root_path=data_root_path, config=config)

    # TODO: Split dataset for different recording sessions
    train_subset, val_subset, test_subset = random_split(dataset=dataset, lengths=ratios)

    print(f"training dataset length: {len(train_subset)}")

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_subset, batch_size=8, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
