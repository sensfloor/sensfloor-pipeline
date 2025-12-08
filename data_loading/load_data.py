from pathlib import Path

import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, random_split

from data_loading.sensfloor_dataset import DatasetConfig, SensfloorPosesDataset

DATA_PATH = Path("./data")


def load_single_recording(folder: Path, config: DatasetConfig) -> SensfloorPosesDataset:
    poses_df = pd.read_csv(folder / "video_poses.csv")
    readout_df = pd.read_csv(folder / "sensfloor_readout.csv")
    return SensfloorPosesDataset(
        poses_df=poses_df,
        sensfloor_readout_df=readout_df,
        config=config,
    )


def load_data(config: DatasetConfig) -> ConcatDataset[SensfloorPosesDataset]:
    folders = [folder for folder in DATA_PATH.iterdir() if folder.is_dir()]
    datasets = [load_single_recording(folder, config) for folder in folders]
    return ConcatDataset(datasets)


def train_val_test_split(
    ratios: tuple[float, float, float],
    config: DatasetConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if sum(ratios) != 1.0:
        message = "Splitting ratios don't add up to 1!"
        raise RuntimeError(message)

    dataset = load_data(config=config)

    # TODO: Split dataset for different recording sessions
    train_subset, val_subset, test_subset = random_split(dataset=dataset, lengths=ratios)

    train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_subset, batch_size=8, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
