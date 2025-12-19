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
    normalize_signals: bool = False


def normalize_roi(roi: torch.Tensor, idle_floor_value: int) -> torch.Tensor:
    normalizes_roi = roi - idle_floor_value
    return normalizes_roi / normalizes_roi.max()


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


@dataclass
class DetailedSensfloorPosesData:
    frame_number: int
    floor: RoIFloor
    untransformed_roi_tensor: torch.Tensor
    transformed_roi_tensor: torch.Tensor
    label_tensor: torch.Tensor


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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.get_detailed_data(index)
        return data.transformed_roi_tensor, data.label_tensor

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
            transformed_roi_tensor = normalize_roi(untransformed_roi_tensor, self.config.floor_config.idle_field_value)

        # Get pose
        label = self.poses_df[self.poses_df["frame"] == frame_number].drop(columns=["frame"]).to_numpy()[0]
        label_tensor = torch.Tensor(label)

        return DetailedSensfloorPosesData(
            frame_number,
            floor,
            untransformed_roi_tensor,
            transformed_roi_tensor,
            label_tensor,
        )


def load_single_dataset(data_path: Path, config: DatasetConfig) -> SensfloorPosesDataset:
    poses_df = pd.read_csv(data_path / "video_poses.csv")
    readout_df = pd.read_csv(data_path / "sensfloor_readout.csv")
    return SensfloorPosesDataset(
        poses_df=poses_df,
        sensfloor_readout_df=readout_df,
        config=config,
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
