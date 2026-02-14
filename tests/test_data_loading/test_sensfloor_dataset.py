from pathlib import Path

import pandas as pd
import torch

from src.data_loading.roi_floor import RoIFloorConfig
from src.training.dataset.dataset_utils import DatasetConfig, normalize_roi
from src.training.dataset.sensfloor_dataset import SensfloorPosesDataset

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

test_sensfloor_readout_path = current_dir / "data" / "test_sensfloor_readout.csv"
test_poses_path = current_dir / "data" / "test_video_poses.csv"


def test_dataset_length():
    sensfloor_readout_df = pd.read_csv(test_sensfloor_readout_path)
    poses_df = pd.read_csv(test_poses_path)
    expected_dataset_length = 9

    floor_config = RoIFloorConfig(x_size=10, y_size=6, history_maxlen=4, roi_size=3)
    config = DatasetConfig(floor_config=floor_config)
    dataset = SensfloorPosesDataset(poses_df, sensfloor_readout_df, config)

    assert len(dataset) == expected_dataset_length


def test_getitem_with_one_signal():
    sensfloor_readout_df = pd.read_csv(test_sensfloor_readout_path)
    poses_df = pd.read_csv(test_poses_path)

    floor_config = RoIFloorConfig(x_size=10, y_size=6, history_maxlen=10, roi_size=3)
    config = DatasetConfig(floor_config=floor_config)
    dataset = SensfloorPosesDataset(poses_df, sensfloor_readout_df, config)

    roi_history, label = dataset[0]

    # 10 frames, 12x12 due to kernel size 3
    assert roi_history.shape == torch.Size([10, 12, 12])

    # 33 keypoints with x, y and z coordinates
    assert label.shape == torch.Size([33 * 3])


def test_getitem_with_multiple_signals():
    sensfloor_readout_df = pd.read_csv(test_sensfloor_readout_path)
    poses_df = pd.read_csv(test_poses_path)

    floor_config = RoIFloorConfig(x_size=10, y_size=6, history_maxlen=10, roi_size=3)
    config = DatasetConfig(floor_config=floor_config)
    dataset = SensfloorPosesDataset(poses_df, sensfloor_readout_df, config)

    roi_history, label = dataset[-1]

    # 10 frames, 12x12 due to kernel size 3
    assert roi_history.shape == torch.Size([10, 12, 12])

    # 33 keypoints with x, y and z coordinates
    assert label.shape == torch.Size([33 * 3])


def test_normalize_roi_history_1():
    roi = torch.ones((1, 12, 12)) * 127
    roi[0, 10, 10] = 200
    roi = normalize_roi(roi, 127, normalize_to_max=True)
    assert roi[0, 10, 10] == 1
    assert roi.sum() == 1


def test_normalize_roi_history_10():
    roi = torch.ones((10, 12, 12)) * 127
    roi[0, :, :] = 177
    roi[7, 1, 1] = 227
    roi = normalize_roi(roi, 127, normalize_to_max=True)

    assert roi[0, 3, 3] == 0.5  # noqa: PLR2004
    assert roi[7, 1, 1] == 1
    assert roi[5, :, :].sum() == 0
