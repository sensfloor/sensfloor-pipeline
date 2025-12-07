from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sensfloor_dataset import SensfloorPosesDataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

DATA_PATH = Path("./data")


@dataclass
class DatasetConfig:
    history_maxlen: int
    floor_size_x: int
    floor_size_y: int
    roi_size: int


def load_data(config: DatasetConfig) -> Dataset:
    folders = [folder for folder in DATA_PATH.iterdir() if folder.isdir()]
    datasets = []
    for folder in folders:
        poses_df = pd.read_csv(folder / "video_poses.csv")
        readout_df = pd.read_csv(folder / "sensfloor_readout.csv")
        datasets.append(
            SensfloorPosesDataset(
                poses_df=poses_df,
                sensfloor_readout_df=readout_df,
                history_maxlen=config.history_maxlen,
                floor_size_x=config.floor_size_x,
                floor_size_y=config.floor_size_y,
                roi_size=config.roi_size,
            ),
        )
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
