from pathlib import Path

import pandas as pd
import torch

from data_loading.sensfloor_dataset import SensfloorPosesDataset

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

test_sensfloor_readout_path = current_dir / "data" / "test_sensfloor_readout.csv"
test_poses_path = current_dir / "data" / "test_video_poses.csv"


def test_dataset_length():
    sensfloor_readout_df = pd.read_csv(test_sensfloor_readout_path)
    poses_df = pd.read_csv(test_poses_path)

    dataset = SensfloorPosesDataset(poses_df, sensfloor_readout_df, 10, 6, 4, 3)

    assert len(dataset) == 9


def test_getitem_with_one_signal():
    sensfloor_readout_df = pd.read_csv(test_sensfloor_readout_path)
    poses_df = pd.read_csv(test_poses_path)

    dataset = SensfloorPosesDataset(poses_df, sensfloor_readout_df, 10, 6, 4, 3)

    roi_history, label = dataset[0]

    # 10 frames, 12x12 due to kernel size 3
    assert roi_history.shape == torch.Size([10, 12, 12])

    # 33 keypoints with x, y and z coordinates
    assert label.shape == torch.Size([33 * 3])


def test_getitem_with_multiple_signals():
    sensfloor_readout_df = pd.read_csv(test_sensfloor_readout_path)
    poses_df = pd.read_csv(test_poses_path)

    dataset = SensfloorPosesDataset(poses_df, sensfloor_readout_df, 10, 6, 4, 3)

    roi_history, label = dataset[-1]

    # 10 frames, 12x12 due to kernel size 3
    assert roi_history.shape == torch.Size([10, 12, 12])

    # 33 keypoints with x, y and z coordinates
    assert label.shape == torch.Size([33 * 3])
