from enum import Enum
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data_loading.roi_floor import RoIFloorConfig
from definitions import TRAIN_DATA_PATH
from training.dataset.dataset_utils import (
    DatasetConfig,
)
from training.dataset.lstm_dataset import LSTMDataset
from training.dataset.sensfloor_dataset import SensfloorPosesDataset


class DatasetType(Enum):
    HISTORY = "history"
    SEQUENCE = "sequence"

def load_single_dataset(data_path: Path, config: DatasetConfig, dataset_type: DatasetType, return_detailed=False) -> SensfloorPosesDataset:
    poses_df = pd.read_csv(data_path / "video_poses.csv")
    readout_df = pd.read_csv(data_path / "sensfloor_readout.csv")

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

    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        # Sequences (batch, different_seq_len, features)
        sequences, labels = zip(*batch)

        # (batch, max_sequence_len, features)
        padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        lengths = torch.tensor([len(seq) for seq in sequences]) # TODO remove this debugging value
        print(f"padded all sequences in batch to shape {padded_seqs.shape} before sequences in batch had lengths: {lengths}")
        return padded_seqs, labels

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader