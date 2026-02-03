from pathlib import Path

import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Subset

from definitions import VIDEO_FILENAME, READOUT_FILENAME
from training.configs import DatasetType
from training.dataset.dataset_utils import (
    DatasetConfig,
)
from training.dataset.lstm_dataset import LSTMDataset, add_0_padding
from training.dataset.sensfloor_dataset import SensfloorPosesDataset


def load_single_dataset(data_path: Path, config: DatasetConfig, dataset_type: DatasetType,
                        return_detailed=False) -> SensfloorPosesDataset:
    poses_df = pd.read_csv(data_path / VIDEO_FILENAME)
    readout_df = pd.read_csv(data_path / READOUT_FILENAME)

    if dataset_type == DatasetType.HISTORY:
        return SensfloorPosesDataset(
            poses_df=poses_df,
            sensfloor_readout_df=readout_df,
            config=config,
            return_detailed=return_detailed,
        )
    return LSTMDataset(
        poses_df=poses_df,
        sensfloor_readout_df=readout_df,
        config=config,
        return_detailed=return_detailed,
    )


def load_all_datasets(
        data_root_path: Path,
        config: DatasetConfig,
        ratios: tuple[float, float, float],
        dataset_type: DatasetType
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
        dataset = load_single_dataset(folder, config, dataset_type)
        train_count = int(ratios[0] * len(dataset))
        val_count = int(ratios[1] * len(dataset))

        indices = list(range(len(dataset)))
        train_idx = indices[:train_count]
        val_idx = indices[train_count: train_count + val_count]
        test_idx = indices[train_count + val_count:]

        train_datasets.append(Subset(dataset, train_idx))
        val_datasets.append(Subset(dataset, val_idx))
        test_datasets.append(Subset(dataset, test_idx))

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), ConcatDataset(test_datasets)


def train_val_test_split(
        data_root_path: Path,
        ratios: tuple[float, float, float],
        config: DatasetConfig,
        batch_size: int,
        dataset_type: DatasetType
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = load_all_datasets(
        data_root_path=data_root_path,
        config=config,
        ratios=ratios,
        dataset_type=dataset_type,
    )

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    loader_args = {}
    if dataset_type == DatasetType.SEQUENCE:
        loader_args["collate_fn"] = add_0_padding

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_args)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_args)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, **loader_args)

    return train_dataloader, val_dataloader, test_dataloader