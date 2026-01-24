from collections.abc import Callable
from pathlib import Path

import torch

from data_loading.floor import PATCH_SIZE
from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloor, RoIFloorConfig
from training.configs import CONFIG_FILE_NAME, ModelType, TrainingConfiguration
from training.models.dataset_utils import normalize_roi
from training.utils import get_device, get_model


def load_model(model_folder: Path) -> tuple[torch.nn.Module, torch.device, ModelType]:
    model_file = model_folder / "best_model.pth"
    config = TrainingConfiguration.load(model_folder / CONFIG_FILE_NAME)

    model = get_model(
        (config.roi_size * PATCH_SIZE, config.roi_size * PATCH_SIZE),
        len(config.landmarks),
        config.roi_history_maxlen,
        config.model_type,
        return_hidden_states=True,
    )

    device = get_device()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)

    return model, device, config.model_type


def load_data_transformations(model_folder: Path) -> Callable[[torch.Tensor], torch.Tensor]:
    config = TrainingConfiguration.load(model_folder / CONFIG_FILE_NAME)

    if config.do_normalize:
        return lambda roi: normalize_roi(roi, 127, config.normalize_to_max)

    return lambda roi: roi


def load_floor(model_folder: Path) -> tuple[RoIFloor, RoIFloorConfig]:
    config = TrainingConfiguration.load(model_folder / CONFIG_FILE_NAME)

    floor_config = RoIFloorConfig(
        x_size=6,
        y_size=4,
        history_maxlen=config.roi_history_maxlen,
        roi_size=config.roi_size,
    )
    return RoIFloor(floor_config), floor_config


def load_pose_landmark_mapping(model_folder: Path) -> list[PoseLandmark]:
    config = TrainingConfiguration.load(model_folder / CONFIG_FILE_NAME)
    return sorted(config.landmarks, key=lambda x: x.value)
