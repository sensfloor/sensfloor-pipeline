import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from data_loading.roi_floor import create_roi_floor, RoIFloorConfig
from training.dataset.dataset_utils import DatasetConfig, normalize_roi, rotate_pose, rotate_roi, drop_landmarks, \
    remove_noise_messages, get_sequences, DetailedSensfloorPosesData


class LSTMDataset(Dataset):
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

        self.sequences = get_sequences(
            sensfloor_readout=filtered_readout,
            poses=poses_df,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> DetailedSensfloorPosesData | tuple[torch.Tensor, torch.Tensor]:
        data = self.get_detailed_data(index)
        if self.return_detailed:
            return data
        return data.transformed_roi_tensor, data.transformed_label_tensor

    def get_detailed_data(self, index: int) -> DetailedSensfloorPosesData:
        # Get signal history
        frame_number, history_len = self.sequences[index]
        old = self.config.floor_config
        floor = create_roi_floor(RoIFloorConfig(history_maxlen=history_len, x_size=old.x_size, y_size=old.y_size, roi_size=old.roi_size), self.sensfloor_readout_df, frame_number)
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


def load_single_dataset(data_path: Path, config: DatasetConfig, return_detailed=False) -> LSTMDataset:
    poses_df = pd.read_csv(data_path / "video_poses.csv")
    readout_df = pd.read_csv(data_path / "sensfloor_readout.csv")
    return LSTMDataset(
        poses_df=poses_df,
        sensfloor_readout_df=readout_df,
        config=config,
        return_detailed = return_detailed,
    )


def load_all_datasets(data_root_path: Path, config: DatasetConfig) -> ConcatDataset[LSTMDataset]:
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
