import json
import random
from enum import Enum, IntEnum
from pathlib import Path
from typing import TypedDict, Literal

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

kept_landmarks_default = [l for l in PoseLandmark if not l in drop_landmarks_default]

drop_feet_landmarks = [PoseLandmark.LEFT_HEEL, PoseLandmark.RIGHT_HEEL, PoseLandmark.LEFT_FOOT_INDEX,
                       PoseLandmark.RIGHT_FOOT_INDEX]


class ModelType(Enum):
    CNN = 0,
    CNN_EFFICIENT = 1,
    CNN_MAX_POOL = 2,
    CNN_RELU_LAST = 3,
    CNN_NO_BATCHNORM = 4,
    CNN_LSTM = 5,
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
    test_path: Path
    model_name: str


ALL_CONFIGS: list[HyperParams] = [
    {
        "epochs": 40,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.79,
            0.2,
            0.01,
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
        "kept_landmarks": [l for l in kept_landmarks_default if not l in drop_feet_landmarks],
        "model_type": ModelType.CNN,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "test_path": ROOT_PATH / "data_testing" / "2025-12-09_15-58-52-line-justin",
        "model_name": "Rotate data",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.79,
            0.2,
            0.01,
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
        "kept_landmarks": [l for l in kept_landmarks_default if not l in drop_feet_landmarks],
        "model_type": ModelType.CNN_LSTM_EFFICIENT,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "test_path": ROOT_PATH / "data_testing" / "2025-12-09_15-58-52-line-justin",
        "model_name": "CNN LSTM Efficient",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.79,
            0.2,
            0.01,
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
        "kept_landmarks": [l for l in kept_landmarks_default if not l in drop_feet_landmarks],
        "model_type": ModelType.CNN_MAX_POOL,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "test_path": ROOT_PATH / "data_testing" / "2025-12-09_15-58-52-line-justin",
        "model_name": "max pool",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.79,
            0.2,
            0.01,
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
        "kept_landmarks": [l for l in kept_landmarks_default if not l in drop_feet_landmarks],
        "model_type": ModelType.CNN,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "test_path": ROOT_PATH / "data_testing" / "2025-12-09_15-58-52-line-justin",
        "model_name": "test model configs",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.79,
            0.2,
            0.01,
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
        "kept_landmarks": [l for l in kept_landmarks_default if not l in drop_feet_landmarks],
        "model_type": ModelType.CNN_EFFICIENT,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "test_path": ROOT_PATH / "data_testing" / "2025-12-09_15-58-52-line-justin",
        "model_name": "efficient",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.79,
            0.2,
            0.01,
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
        "kept_landmarks": [l for l in kept_landmarks_default if not l in drop_feet_landmarks],
        "model_type": ModelType.CNN_RELU_LAST,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "test_path": ROOT_PATH / "data_testing" / "2025-12-09_15-58-52-line-justin",
        "model_name": "CNN relu last",
    },
    {
        "epochs": 40,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": random.randint(0, 1_000_000),
        "split_ratios": (
            0.79,
            0.2,
            0.01,
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
        "kept_landmarks": [l for l in kept_landmarks_default if not l in drop_feet_landmarks],
        "model_type": ModelType.CNN_NO_BATCHNORM,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 7,
        "amplify_link_loss": 0.1,
        "mse_loss": "mean",
        "test_path": ROOT_PATH / "data_testing" / "2025-12-09_15-58-52-line-justin",
        "model_name": "No batchnorm",
    },
]


def get_hyper_param_configs():
    for config in ALL_CONFIGS:
        if len(config["model_name"]) == 0:
            print(f"No Model name configured, taking seed {config['seed']} as name")
            config["model_name"] = f"{config['seed']}"

    names = [config["model_name"] for config in ALL_CONFIGS]
    if len(names) != len(set(names)):
        print("duplicate model names, adding index")
        for i, config in enumerate(ALL_CONFIGS):
            config["model_name"] += f"_{i}"

    names = [config["model_name"] for config in ALL_CONFIGS]
    print(f"running these configs: {names}")
    return ALL_CONFIGS


PROJECT_NAME = "sensfloor_cairo_6"


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
    with (folder_path / file_name).open('w') as f:
        json.dump(params, f, cls=HyperParamsEncoder, indent=4)
    print(f"Hyperparams saved to {folder_path}")


def load_hyperparams(folder_path: Path, file_name: str = CONFIG_FILE_NAME) -> HyperParams:
    """
    Loads HyperParams from JSON and restores specific Python types
    (Path, Tuple, Enums) that JSON converts to basic types.
    """
    with (folder_path / file_name).open('r') as f:
        data = json.load(f)

    # --- Handle "complex" objects ---
    if "test_path" in data:
        data["test_path"] = Path(data["test_path"])

    if "split_ratios" in data:
        data["split_ratios"] = tuple(data["split_ratios"])

    if "kept_landmarks" in data:
        data["kept_landmarks"] = [PoseLandmark[name] for name in data["kept_landmarks"]]

    if "model_type" in data:
        data["model_type"] = ModelType[data["model_type"]]

    return data


# --- Usage Example ---
if __name__ == "__main__":
    # Create dummy data
    params: HyperParams = {
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32,
        "seed": 42,
        "split_ratios": (0.7, 0.2, 0.1),
        "patch_width": 64,
        "floor_x_size": 128,
        "floor_y_size": 128,
        "roi_history_maxlen": 10,
        "roi_size": 256,
        "do_normalize": True,
        "normalize_to_max": False,
        "rotate_data": True,
        "training_data_folders": ["/data/set1", "/data/set2"],
        "kept_landmarks": [PoseLandmark.NOSE, PoseLandmark.LEFT_EYE],
        "model_type": ModelType.CNN_LSTM,
        "scheduler_patience": 5,
        "scheduler_min_lr": 1e-6,
        "scheduler_factor": 0.1,
        "trainer_patience": 10,
        "amplify_link_loss": 1.5,
        "mse_loss": "mean",
        "test_path": Path("./tests"),
        "model_name": "pose_v1"
    }

    # Save
    save_hyperparams(params, ROOT_PATH)

    # Load
    loaded_params = load_hyperparams(ROOT_PATH)

    # Verify complex types were restored correctly
    print(f"Restored test_path type: {type(loaded_params['test_path'])}")  # <class 'pathlib.Path'>
    print(f"Restored split_ratios type: {type(loaded_params['split_ratios'])}")  # <class 'tuple'>
    print(f"Restored Enum: {loaded_params['kept_landmarks'][0]}")  # PoseLandmark.NOSE