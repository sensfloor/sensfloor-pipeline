from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_PATH = Path("../data")

folders = [directory for directory in DATA_PATH.iterdir() if directory.is_dir()]

poses_df = pd.read_csv(folders[0] / "poses.csv")
sensfloor_readout_df = pd.read_csv(folders[0] / "sensfloor_readout.csv")


class SensfloorPosesDataset(Dataset):
    def __init__(self, poses_df: pd.DataFrame, sensfloor_readout_df: pd.DataFrame) -> None:
        super().__init__()
        self.poses_df = poses_df
        self.sensfloor_readout_df = sensfloor_readout_df
        self.rois = []

        frames = self.sensfloor_readout_df.groupby(["frame"])

    def __len__(self) -> int:
        return -1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__getitem__(index)


print(poses_df.head(1))
print(sensfloor_readout_df.head(1))
