import numpy as np

from data_loading.floor import PATCH_SIZE
from tracking.clustering import calculate_activation_cluster_means
from tracking.kalman_filter import SensfloorKalmanFilter


class PersonTracker:
    def __init__(self, fps: float, idle_field_value: int, filter_reset_threshold: int) -> None:
        self.kalman_filter = SensfloorKalmanFilter(fps)
        self.idle_field_value = idle_field_value
        self.frames_without_position = 0
        self.filter_reset_threshold = filter_reset_threshold

    def track(self, floor: np.ndarray) -> np.ndarray:
        self.kalman_filter.predict()
        detected_position = calculate_activation_cluster_means(floor, self.idle_field_value)

        if len(detected_position) == 0:
            self.frames_without_position += 1
            return self.kalman_filter.x / PATCH_SIZE

        if self.frames_without_position > self.filter_reset_threshold:
            self.kalman_filter.reset()

        self.kalman_filter.update(detected_position)
        self.frames_without_position = 0

        return self.kalman_filter.x / PATCH_SIZE
