import numpy as np
from filterpy.kalman import KalmanFilter

FILTER_OPTIMIZED_ACTIVE_FIELD_MIN_VALUE = 145


class SensfloorKalmanFilter:
    def __init__(self, fps: float) -> None:
        self.dt = 1 / fps
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.reset()

    def reset(self) -> None:
        self.kf.P = np.eye(4) * 10
        self.kf.x = np.array([0, 0, 0, 0], dtype=np.float64)  # (Initial) State estimate
        self.kf.F = np.array(  # State transition matrix
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.kf.Q = np.eye(4) * 0.0003  # Transition noise
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)  # State -> Measurement mapping
        self.kf.R = np.eye(2) * 0.01  # Measurement noise

    def predict(self) -> None:
        self.kf.predict()

    def update(self, position: np.ndarray) -> None:
        self.last_x = self.x
        self.kf.update(position)

    @property
    def x(self) -> np.ndarray:
        return self.kf.x[:2]
