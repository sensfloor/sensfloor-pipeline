from __future__ import annotations

import random
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from src.data_loading.pose_landmark import PoseLandmark
from src.data_loading.roi_offset_strategy import OffsetStrategy
from src.definitions import HOLD_OUT_DATA_PATH, MODELS_FOLDER_PATH, TRAIN_DATA_PATH

CONFIG_FILE_NAME = "config.json"
PROJECT_NAME = "sensfloor_cairo_12_new_metrics"
PROJECT_GROUP = "bigger_model"

training_folders = [directory.name for directory in TRAIN_DATA_PATH.iterdir() if directory.is_dir()]
hold_out_folders = [directory.name for directory in HOLD_OUT_DATA_PATH.iterdir() if directory.is_dir()]

landmark_weights: dict[PoseLandmark, float] = {
    PoseLandmark.NOSE : 1.0,
    PoseLandmark.LEFT_SHOULDER : 1.0,
    PoseLandmark.RIGHT_SHOULDER : 1.0,
    PoseLandmark.LEFT_ELBOW : 1.0,
    PoseLandmark.RIGHT_ELBOW : 1.0,
    PoseLandmark.LEFT_WRIST : 1.0,
    PoseLandmark.RIGHT_WRIST : 1.0,
    PoseLandmark.LEFT_HIP : 1.0,
    PoseLandmark.RIGHT_HIP : 1.0,
    PoseLandmark.LEFT_KNEE : 1.0,
    PoseLandmark.RIGHT_KNEE : 1.0,
    PoseLandmark.LEFT_ANKLE : 1.0,
    PoseLandmark.RIGHT_ANKLE : 1.0,
}

def get_weighted_feet(weight: float) -> dict[PoseLandmark, float]:
    return {
    PoseLandmark.NOSE : 1.0,
    PoseLandmark.LEFT_SHOULDER : 1.0,
    PoseLandmark.RIGHT_SHOULDER : 1.0,
    PoseLandmark.LEFT_ELBOW : 1.0,
    PoseLandmark.RIGHT_ELBOW : 1.0,
    PoseLandmark.LEFT_WRIST : 1.0,
    PoseLandmark.RIGHT_WRIST : 1.0,
    PoseLandmark.LEFT_HIP : 1.0,
    PoseLandmark.RIGHT_HIP : 1.0,
    PoseLandmark.LEFT_KNEE : weight,
    PoseLandmark.RIGHT_KNEE : weight,
    PoseLandmark.LEFT_ANKLE : weight,
    PoseLandmark.RIGHT_ANKLE : weight,
}


class ModelType(Enum):
    CNN = 0
    CNN_EFFICIENT = 1
    CNN_MAX_POOL = 2
    CNN_RELU_LAST = 3
    CNN_NO_BATCHNORM = 4
    CNN_LSTM = 5
    CNN_LSTM_EFFICIENT = 6

def is_lstm(model: ModelType) -> bool:
    match model:
        case ModelType.CNN_LSTM | ModelType.CNN_LSTM_EFFICIENT:
            return True
        case (ModelType.CNN | ModelType.CNN_EFFICIENT | 
              ModelType.CNN_MAX_POOL | ModelType.CNN_RELU_LAST | ModelType.CNN_NO_BATCHNORM):
            return False
    
    message = "model_type not defined"
    raise ValueError(message)


class DatasetType(Enum):
    HISTORY = "history"
    SEQUENCE = "sequence"


class TrainingConfiguration(BaseModel):
    # Pydantic configuration
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
    )

    # Training Params
    epochs: int
    learning_rate: float
    batch_size: int
    seed: int
    split_ratios: tuple[float, float, float]

    # Model/Data Params
    patch_width: int
    active_field_min_value: int
    floor_x_size: int
    floor_y_size: int
    roi_history_maxlen: int
    roi_size: int | None
    roi_offset_strategy: OffsetStrategy
    do_normalize: bool
    normalize_to_max: bool
    rotate_data: bool
    remove_noise: bool

    training_data_folders: list[str]
    landmarks: list[PoseLandmark]
    landmark_weights: dict[PoseLandmark, float] = landmark_weights
    model_type: ModelType
    dataset_type: DatasetType = DatasetType.HISTORY

    # Optimizer/Trainer Params
    scheduler_patience: int
    scheduler_min_lr: float
    scheduler_factor: float
    trainer_patience: int
    amplify_link_loss: float

    # Testing
    hold_out_data_folder: list[str]
    model_name: str

    @field_serializer("model_type")
    def serialize_model_type(self, model_type: ModelType) -> str:
        return model_type.name

    @field_validator("model_type", mode="before")
    @classmethod
    def parse_model_type(cls, v: Any) -> Any:  # noqa: ANN401
        if isinstance(v, str):
            return ModelType[v]
        return v

    @field_serializer("landmarks")
    def serialize_landmarks(self, landmarks: list[PoseLandmark]) -> list[str]:
        return [landmark.name for landmark in landmarks]

    @field_validator("landmarks", mode="before")
    @classmethod
    def parse_landmarks(cls, v: Any) -> Any:  # noqa: ANN401
        if isinstance(v, list):
            return [PoseLandmark[name] if isinstance(name, str) else name for name in v]
        return v
    
    
    @field_serializer("landmark_weights")
    def serialize_landmark_weights(self, landmark_weights: dict[PoseLandmark, float]) -> dict[str, float]:
        return {landmark.name: weight for (landmark, weight) in landmark_weights.items()}
    

    def save(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(self.model_dump_json(indent=4), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        if not path.exists():
            message = f"Configuration with path {path} not found"
            raise FileNotFoundError(message)

        json_data = path.read_text(encoding="utf-8")
        return cls.model_validate_json(json_data)


_BASE_CONFIG = TrainingConfiguration(
    epochs=40,
    learning_rate=1e-4,
    batch_size=32,
    seed=random.randint(0, 1_000_000),
    split_ratios=(0.8, 0.1, 0.1),
    patch_width=4,
    active_field_min_value=140,
    floor_x_size=6,
    floor_y_size=4,
    roi_history_maxlen=50,
    roi_size=4,
    roi_offset_strategy=OffsetStrategy.EXHAUSTIVE,
    do_normalize=True,
    normalize_to_max=False,
    rotate_data=False,
    remove_noise=True,
    training_data_folders=training_folders,
    landmarks=list(landmark_weights.keys()),
    landmark_weights=get_weighted_feet(5),
    model_type=ModelType.CNN_LSTM_EFFICIENT,
    dataset_type=DatasetType.HISTORY,
    scheduler_patience=3,
    scheduler_min_lr=1e-6,
    scheduler_factor=0.1,
    trainer_patience=5,
    amplify_link_loss=0.1,
    hold_out_data_folder=hold_out_folders,
    model_name="base_config_weighted",
)

_ALL_CONFIGS = [
    _BASE_CONFIG.model_copy(update={"model_name": "model_type_CNNLSTM", "model_type": ModelType.CNN_LSTM}),
    _BASE_CONFIG,
]


def get_hyper_param_configs() -> list[TrainingConfiguration]:
    for config in _ALL_CONFIGS:
        model_folder = MODELS_FOLDER_PATH / config.model_name
        if model_folder.is_dir():
            message = f"Folder for model {config.model_name} already exists."
            raise RuntimeError(message)

    names = [config.model_name for config in _ALL_CONFIGS]
    if len(names) != len(set(names)):
        message = "Duplicate model names in configurations."
        raise RuntimeError(message)

    return _ALL_CONFIGS
