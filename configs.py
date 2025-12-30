import random
from pathlib import Path
from typing import TypedDict, Literal

from data_loading.pose_landmark import PoseLandmark

data_root = Path("./data")
training_folders = [f.name for f in data_root.iterdir() if f.is_dir()]

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

class HyperParams(TypedDict):
    # Training Params
    epochs: int
    learning_rate: float
    batch_size: int
    seed: int
    split_ratios: tuple[float, float, float]

    # Model/Data Params
    patch_width: int
    roi_x_size: int
    roi_y_size: int
    roi_history_maxlen: int
    roi_size: int
    do_normalize: bool
    normalize_to_max: bool
    rotate_data: bool

    training_data_folders: list[str]
    dropped_landmarks: list[PoseLandmark]

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


def get_hyper_param_configs():
    all_configs: list[HyperParams] = [
        {
            "epochs": 30,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "seed": random.randint(0, 1_000_000),
            "split_ratios": (
                0.79,
                0.2,
                0.01,
            ),

            "patch_width": 4,
            "roi_x_size": 6,
            "roi_y_size": 4,
            "roi_history_maxlen": 25,
            "roi_size": 3,
            "do_normalize": True,
            "normalize_to_max": False,
            "rotate_data": False,

            "training_data_folders": training_folders,
            "dropped_landmarks": drop_landmarks_default,

            "scheduler_patience": 3,
            "scheduler_min_lr": 1e-6,
            "scheduler_factor": 0.1,
            "trainer_patience": 7,
            "amplify_link_loss": 0.1,
            "mse_loss": "mean",

            "test_path": Path("./data_testing/2025-12-09_15-58-52-line-justin"),
            "model_name": "best config combination 2",
        },

        {
            "epochs": 30,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "seed": random.randint(0, 1_000_000),
            "split_ratios": (
                0.79,
                0.2,
                0.01,
            ),

            "patch_width": 4,
            "roi_x_size": 6,
            "roi_y_size": 4,
            "roi_history_maxlen": 25,
            "roi_size": 3,
            "do_normalize": True,
            "normalize_to_max": False,
            "rotate_data": True,

            "training_data_folders": training_folders,
            "dropped_landmarks": drop_landmarks_default,

            "scheduler_patience": 3,
            "scheduler_min_lr": 1e-6,
            "scheduler_factor": 0.1,
            "trainer_patience": 7,
            "amplify_link_loss": 0.1,
            "mse_loss": "mean",

            "test_path": Path("./data_testing/2025-12-09_15-58-52-line-justin"),
            "model_name": "rotate data",
        },

        {
            "epochs": 30,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "seed": random.randint(0, 1_000_000),
            "split_ratios": (
                0.79,
                0.2,
                0.01,
            ),

            "patch_width": 4,
            "roi_x_size": 6,
            "roi_y_size": 4,
            "roi_history_maxlen": 25,
            "roi_size": 3,
            "do_normalize": True,
            "normalize_to_max": False,
            "rotate_data": False,

            "training_data_folders": training_folders,
            "dropped_landmarks": drop_landmarks_default,

            "scheduler_patience": 3,
            "scheduler_min_lr": 1e-6,
            "scheduler_factor": 0.1,
            "trainer_patience": 7,
            "amplify_link_loss": 0.1,
            "mse_loss": "mean",

            "test_path": Path("./data_testing/2025-12-09_15-58-52-line-justin"),
            "model_name": "lr 5e-4 second run",
        },

        {
            "epochs": 30,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "seed": random.randint(0, 1_000_000),
            "split_ratios": (
                0.79,
                0.2,
                0.01,
            ),

            "patch_width": 4,
            "roi_x_size": 6,
            "roi_y_size": 4,
            "roi_history_maxlen": 25,
            "roi_size": 3,
            "do_normalize": True,
            "normalize_to_max": False,
            "rotate_data": False,

            "training_data_folders": training_folders,
            "dropped_landmarks": drop_landmarks_default,

            "scheduler_patience": 3,
            "scheduler_min_lr": 1e-6,
            "scheduler_factor": 0.1,
            "trainer_patience": 7,
            "amplify_link_loss": 0.3,
            "mse_loss": "mean",

            "test_path": Path("./data_testing/2025-12-09_15-58-52-line-justin"),
            "model_name": "link_loss 0.3",
        },

        {
            "epochs": 30,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "seed": random.randint(0, 1_000_000),
            "split_ratios": (
                0.79,
                0.2,
                0.01,
            ),

            "patch_width": 4,
            "roi_x_size": 6,
            "roi_y_size": 4,
            "roi_history_maxlen": 25,
            "roi_size": 3,
            "do_normalize": True,
            "normalize_to_max": False,
            "rotate_data": False,

            "training_data_folders": training_folders,
            "dropped_landmarks": drop_landmarks_default,

            "scheduler_patience": 3,
            "scheduler_min_lr": 1e-6,
            "scheduler_factor": 0.1,
            "trainer_patience": 7,
            "amplify_link_loss": 0.5,
            "mse_loss": "mean",

            "test_path": Path("./data_testing/2025-12-09_15-58-52-line-justin"),
            "model_name": "link_loss 0.5",
        },

        {
            "epochs": 30,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "seed": random.randint(0, 1_000_000),
            "split_ratios": (
                0.79,
                0.2,
                0.01,
            ),

            "patch_width": 4,
            "roi_x_size": 6,
            "roi_y_size": 4,
            "roi_history_maxlen": 25,
            "roi_size": 3,
            "do_normalize": True,
            "normalize_to_max": False,
            "rotate_data": False,

            "training_data_folders": training_folders,
            "dropped_landmarks": drop_landmarks_default,

            "scheduler_patience": 3,
            "scheduler_min_lr": 1e-6,
            "scheduler_factor": 0.1,
            "trainer_patience": 7,
            "amplify_link_loss": 0.1,
            "mse_loss": "mean",

            "test_path": Path("./data_testing/2025-12-09_15-58-52-line-justin"),
            "model_name": "batch_size 32",
        },
    ]

    for config in all_configs:
        if len(config["model_name"]) == 0:
            print(f"No Model name configured, taking seed {config['seed']} as name")
            config["model_name"] = f"{config['seed']}"

    names = [config["model_name"] for config in all_configs]
    if len(names) != len(set(names)):
        print(f"duplicate model names, adding index")
        for i, config in enumerate(all_configs):
            config["model_name"] += f"_{i}"

    names = [config["model_name"] for config in all_configs]
    print(f"running these configs: {names}")
    return all_configs


PROJECT_NAME = "sensfloor_cairo_3"
