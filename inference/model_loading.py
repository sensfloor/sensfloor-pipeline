from collections.abc import Callable
from pathlib import Path

import torch

from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloor, RoIFloorConfig
from training.configs import load_hyperparams
from training.models.dataset_utils import normalize_roi
from training.utils import get_device, get_model


def load_model(model_folder: Path) -> tuple[torch.nn.Module, torch.device]:
    model_file = model_folder / "best_model.pth"
    config = load_hyperparams(model_folder)
    model = get_model(
        (config["roi_size"] * 4, config["roi_size"] * 4),
        len(config["kept_landmarks"]),
        config["roi_history_maxlen"],
        config["model_type"],
    )

    device = get_device()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)

    return model, device


def load_data_transformations(model_folder: Path) -> Callable[[torch.Tensor], torch.Tensor]:
    config = load_hyperparams(model_folder)

    if config["do_normalize"]:
        return lambda roi: normalize_roi(roi, 127, config["normalize_to_max"])

    return lambda roi: roi


def load_floor(model_folder: Path) -> tuple[RoIFloor, RoIFloorConfig]:
    config = load_hyperparams(model_folder)
    floor_config = RoIFloorConfig(
        x_size=6,
        y_size=4,
        history_maxlen=config["roi_history_maxlen"],
        roi_size=config["roi_size"],
    )
    return RoIFloor(floor_config), floor_config


def load_pose_landmark_mapping(model_folder: Path) -> list[PoseLandmark]:
    config = load_hyperparams(model_folder)
    return sorted(config["kept_landmarks"], key=lambda x: x.value)
