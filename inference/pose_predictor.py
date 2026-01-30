from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from inference.model_loading import load_data_transformations, load_floor, load_model, load_pose_landmark_mapping
from training.configs import ModelType


class JointPrediction(TypedDict):
    joint: str
    x: float
    y: float
    z: float


class PosePredictor:
    def __init__(self, model_folder: Path, num_calls_cache: int) -> None:
        self.model, self.device, self.model_type = load_model(model_folder)
        self.transform_data = load_data_transformations(model_folder)
        self.pose_landmark_mapping = load_pose_landmark_mapping(model_folder)
        self.floor, self.floor_config = load_floor(model_folder)

        print(f"Use model model({self.model_type.name}) on {self.device} to predict poses")

        self.num_calls_cache = num_calls_cache
        self.last_prediction: list[JointPrediction] = []
        self.calls_without_prediction = 0
        self.h_c = None

    def predict(self, positions: np.ndarray, signals: np.ndarray) -> list[JointPrediction]:
        if self.calls_without_prediction > self.num_calls_cache:
            self.last_prediction = []

        self.floor.update(positions, signals)
        roi = self.floor.get_roi()

        if roi is None:
            self.calls_without_prediction += 1
            return self.last_prediction

        x = torch.Tensor(roi.history).unsqueeze(0)
        x = x.to(self.device)
        x = self.transform_data(x)

        # TODO: add logic to reset h_c if there were e.g. 15 frames without signal
        if self.model_type in [ModelType.CNN_LSTM, ModelType.CNN_LSTM_EFFICIENT]:
            outputs, (self.h_c) = self.model(x, self.h_c)
        else:
            outputs = self.model(x)

        joints = outputs.reshape(-1, 3).cpu().tolist()

        self.last_prediction = [
            {"joint": self.pose_landmark_mapping[i].name, "x": joint[0], "y": joint[1], "z": joint[2]}
            for i, joint in enumerate(joints)
        ]
        self.calls_without_prediction = 0

        return self.last_prediction
