import json
import random
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict

from data_loading.pose_landmark import PoseLandmark
from definitions import DATA_PATH, ROOT_PATH

training_folders = [f.name for f in DATA_PATH.iterdir() if f.is_dir()]

drop_landmarks_default = [
    PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_INNER,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE,
    PoseLandmark.RIGHT_EYE_INNER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.MOUTH_RIGHT,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EAR,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_PINKY,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.RIGHT_PINKY,
]

kept_landmarks_default = [l for l in PoseLandmark if l not in drop_landmarks_default]

drop_feet_landmarks = [
    PoseLandmark.LEFT_HEEL,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.RIGHT_FOOT_INDEX,
]


class ModelType(Enum):
    CNN = (0,)
    CNN_EFFICIENT = (1,)
    CNN_MAX_POOL = (2,)
    CNN_RELU_LAST = (3,)
    CNN_NO_BATCHNORM = (4,)
    CNN_LSTM = (5,)
    CNN_LSTM_EFFICIENT = 6


class HyperParams(TypedDict):
    # Training Params
    epochs: int
    learning_rate: float
    batch_size: int
    seed: int
    split_ratios: tuple[float, float, float]

    # Model/Data Params
    patch_width: int
    floor_x_size: int
    floor_y_size: int
    roi_history_maxlen: int
    roi_size: int
    do_normalize: bool
    normalize_to_max: bool
    rotate_data: bool

    training_data_folders: list[str]
    kept_landmarks: list[PoseLandmark]
    model_type: ModelType

    # Optimizer/Trainer Params
    scheduler_patience: int
    scheduler_min_lr: float
    scheduler_factor: float
    trainer_patience: int
    amplify_link_loss: float
    mse_loss: Literal["mean", "sum"]

    # Testing
    create_predictions_path: Path
    model_name: str


ALL_CONFIGS: list[HyperParams] = [
    {
        "epochs": 40,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.8,
            0.1,
            0.1,
        ),
        "patch_width": 4,
        "floor_x_size": 6,
        "floor_y_size": 4,
        "roi_history_maxlen": 25,
        "roi_size": 3,
        "do_normalize": True,
        "normalize_to_max": False,
        "rotate_data": False,
        "training_data_folders": training_folders,
        "kept_landmarks": [l for l in kept_landmarks_default if l not in drop_feet_landmarks],
        "model_type": ModelType.CNN_LSTM,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "create_predictions_path": ROOT_PATH / "data_testing" / "2025-12-16_12-07-42-rikuto-shorts",
        "model_name": "CNN_LSTM",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.8,
            0.1,
            0.1,
        ),
        "patch_width": 4,
        "floor_x_size": 6,
        "floor_y_size": 4,
        "roi_history_maxlen": 25,
        "roi_size": 3,
        "do_normalize": True,
        "normalize_to_max": False,
        "rotate_data": False,
        "training_data_folders": training_folders,
        "kept_landmarks": [l for l in kept_landmarks_default if l not in drop_feet_landmarks],
        "model_type": ModelType.CNN_LSTM_EFFICIENT,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "create_predictions_path": ROOT_PATH / "data_testing" / "2025-12-16_12-07-42-rikuto-shorts",
        "model_name": "CNN_LSTM_EFFICIENT",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.8,
            0.1,
            0.1,
        ),
        "patch_width": 4,
        "floor_x_size": 6,
        "floor_y_size": 4,
        "roi_history_maxlen": 25,
        "roi_size": 3,
        "do_normalize": True,
        "normalize_to_max": False,
        "rotate_data": False,
        "training_data_folders": training_folders,
        "kept_landmarks": [l for l in kept_landmarks_default if l not in drop_feet_landmarks],
        "model_type": ModelType.CNN_EFFICIENT,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "create_predictions_path": ROOT_PATH / "data_testing" / "2025-12-16_12-07-42-rikuto-shorts",
        "model_name": "CNN efficient",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.8,
            0.1,
            0.1,
        ),
        "patch_width": 4,
        "floor_x_size": 6,
        "floor_y_size": 4,
        "roi_history_maxlen": 25,
        "roi_size": 3,
        "do_normalize": True,
        "normalize_to_max": False,
        "rotate_data": False,
        "training_data_folders": training_folders,
        "kept_landmarks": [l for l in kept_landmarks_default if l not in drop_feet_landmarks],
        "model_type": ModelType.CNN_MAX_POOL,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "create_predictions_path": ROOT_PATH / "data_testing" / "2025-12-16_12-07-42-rikuto-shorts",
        "model_name": "CNN_MAX_POOL",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.8,
            0.1,
            0.1,
        ),
        "patch_width": 4,
        "floor_x_size": 6,
        "floor_y_size": 4,
        "roi_history_maxlen": 25,
        "roi_size": 3,
        "do_normalize": True,
        "normalize_to_max": False,
        "rotate_data": False,
        "training_data_folders": training_folders,
        "kept_landmarks": [l for l in kept_landmarks_default if l not in drop_feet_landmarks],
        "model_type": ModelType.CNN_NO_BATCHNORM,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "create_predictions_path": ROOT_PATH / "data_testing" / "2025-12-16_12-07-42-rikuto-shorts",
        "model_name": "CNN_NO_BATCHNORM",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.8,
            0.1,
            0.1,
        ),
        "patch_width": 4,
        "floor_x_size": 6,
        "floor_y_size": 4,
        "roi_history_maxlen": 25,
        "roi_size": 3,
        "do_normalize": True,
        "normalize_to_max": False,
        "rotate_data": False,
        "training_data_folders": training_folders,
        "kept_landmarks": [l for l in kept_landmarks_default if l not in drop_feet_landmarks],
        "model_type": ModelType.CNN_RELU_LAST,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "create_predictions_path": ROOT_PATH / "data_testing" / "2025-12-16_12-07-42-rikuto-shorts",
        "model_name": "CNN_RELU_LAST",
    },
]

MODELS_FOLDER_PATH = ROOT_PATH / "outputs" / "models"


def get_hyper_param_configs():
    for config in ALL_CONFIGS:
        if len(config["model_name"]) == 0:
            print(f"No Model name configured, taking seed {config['seed']} as name")
            config["model_name"] = f"{config['seed']}"
        model_folder = MODELS_FOLDER_PATH / config["model_name"]
        if model_folder.is_dir():
            print(f"Model name exists, taking seed {config['seed']} as name to prevent overwriting")
            config["model_name"] = f"{config['model_name']}_{config['seed']}"

    names = [config["model_name"] for config in ALL_CONFIGS]
    if len(names) != len(set(names)):
        print("duplicate model names, adding index")
        for i, config in enumerate(ALL_CONFIGS):
            config["model_name"] += f"_{i}"

    names = [config["model_name"] for config in ALL_CONFIGS]
    print(f"running these configs: {names}")
    return ALL_CONFIGS


PROJECT_NAME = "sensfloor_cairo_10.1"


class HyperParamsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.name
        return super().default(obj)


CONFIG_FILE_NAME = "config.json"


def save_hyperparams(params: HyperParams, folder_path: Path, file_name: str = CONFIG_FILE_NAME) -> None:
    """Saves HyperParams to a JSON file, handling Enums and Paths."""
    folder_path.mkdir(exist_ok=True, parents=True)
    with (folder_path / file_name).open("w") as f:
        json.dump(params, f, cls=HyperParamsEncoder, indent=4)
    print(f"Hyperparams saved to {folder_path}")


def load_hyperparams(folder_path: Path, file_name: str = CONFIG_FILE_NAME) -> HyperParams:
    """
    Loads HyperParams from JSON and restores specific Python types
    (Path, Tuple, Enums) that JSON converts to basic types.
    """
    with (folder_path / file_name).open("r") as f:
        data = json.load(f)

    # --- Handle "complex" objects ---
    if "create_predictions_path" in data:
        data["create_predictions_path"] = Path(data["create_predictions_path"])

    if "split_ratios" in data:
        data["split_ratios"] = tuple(data["split_ratios"])

    if "kept_landmarks" in data:
        data["kept_landmarks"] = [PoseLandmark[name] for name in data["kept_landmarks"]]

    if "model_type" in data:
        data["model_type"] = ModelType[data["model_type"]]

    return data
