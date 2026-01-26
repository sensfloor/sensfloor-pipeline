import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from data_loading.roi_floor import create_roi_floor
from training.dataset.dataset_utils import (
    DatasetConfig,
    DetailedSensfloorPosesData,
    drop_landmarks,
    get_unique_frames_with_poses,
    normalize_roi,
    remove_noise_messages,
    rotate_pose,
    rotate_roi,
)


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
        return_detailed=return_detailed,
    )


def load_all_datasets(
    data_root_path: Path,
    config: DatasetConfig,
    ratios: tuple[float, float, float],
) -> tuple[
    ConcatDataset[SensfloorPosesDataset],
    ConcatDataset[SensfloorPosesDataset],
    ConcatDataset[SensfloorPosesDataset],
]:
    # TODO: Use Training hyperparams folders
    folders = [folder for folder in data_root_path.iterdir() if folder.is_dir()]

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for folder in folders:
        dataset = load_single_dataset(folder, config)
        train_count = int(ratios[0] * len(dataset))
        val_count = int(ratios[1] * len(dataset))

        indices = list(range(len(dataset)))
        train_idx = indices[:train_count]
        val_idx = indices[train_count : train_count + val_count]
        test_idx = indices[train_count + val_count :]

        train_datasets.append(Subset(dataset, train_idx))
        val_datasets.append(Subset(dataset, val_idx))
        test_datasets.append(Subset(dataset, test_idx))

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), ConcatDataset(test_datasets)


def train_val_test_split(
    data_root_path: Path,
    ratios: tuple[float, float, float],
    config: DatasetConfig,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = load_all_datasets(
        data_root_path=data_root_path,
        config=config,
        ratios=ratios,
    )

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
