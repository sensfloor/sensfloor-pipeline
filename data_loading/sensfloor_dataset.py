import pandas as pd
import torch
from torch.utils.data import Dataset

from data_loading.roi_floor import RoIFloor


class SensfloorPosesDataset(Dataset):
    def __init__(
        self,
        poses_df: pd.DataFrame,
        sensfloor_readout_df: pd.DataFrame,
        history_maxlen: int,
        floor_size_x: int,
        floor_size_y: int,
        roi_size: int,
    ) -> None:
        super().__init__()
        self.poses_df = poses_df
        self.sensfloor_readout_df = sensfloor_readout_df
        self.frames_containing_messages = self.sensfloor_readout_df[
            "frame_number"
        ].unique()
        self.history_maxlen = history_maxlen
        self.floor_size_x = floor_size_x
        self.floor_size_y = floor_size_y
        self.roi_size = roi_size

    def __len__(self) -> int:
        return len(self.frames_containing_messages)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get signal history
        frame_number = self.frames_containing_messages[index]
        floor = self._create_roi_floor(frame_number)
        roi = floor.get_roi()

        if roi is None:
            raise RuntimeError("No region of interest found...")

        roi_tensor = torch.Tensor(roi.history)

        # Get pose
        label = (
            self.poses_df[self.poses_df["frame"] == 10]
            .drop(columns=["frame"])
            .values[0]
        )
        label_tensor = torch.Tensor(label)

        return roi_tensor, label_tensor

    def _create_roi_floor(self, frame_number: int) -> RoIFloor:
        floor = RoIFloor(
            self.floor_size_x, self.floor_size_y, self.history_maxlen, self.roi_size
        )
        earliest_frame_included = frame_number - self.history_maxlen
        for current_frame_number in range(earliest_frame_included, frame_number + 1):
            messages = self.sensfloor_readout_df[
                self.sensfloor_readout_df["frame_number"] == current_frame_number
            ]
            positions = (
                messages[["x", "y"]].values - 1
            ).tolist()  # Subtract 1 from positions because they start at 1,1
            signals = messages[["0", "1", "2", "3", "4", "5", "6", "7"]].values
            floor.update(positions, signals)
        return floor
