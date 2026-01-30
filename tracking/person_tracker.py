import numpy as np

from data_loading.floor import PATCH_SIZE, Floor, FloorConfig
from tracking.clustering import calculate_activation_cluster_means
from tracking.kalman_filter import FILTER_OPTIMIZED_ACTIVE_FIELD_MIN_VALUE, SensfloorKalmanFilter


class PersonTracker:
    def __init__(
        self,
        fps: float,
        filter_reset_threshold: int,
        floor_config: FloorConfig,
    ) -> None:
        self.kalman_filter = SensfloorKalmanFilter(fps)
        self.frames_without_position = 0
        self.filter_reset_threshold = filter_reset_threshold

        if floor_config.active_field_min_value != FILTER_OPTIMIZED_ACTIVE_FIELD_MIN_VALUE:
            warning = (
                "Used kalman filter for person tracker filter is optimized for floor"
                f"active_field_min_value == {FILTER_OPTIMIZED_ACTIVE_FIELD_MIN_VALUE}."
                "Using a different value may result in decreased performance."
            )
            raise Warning(warning)

        self.floor = Floor(floor_config)

    def track(self, positions: np.ndarray, signals: np.ndarray) -> np.ndarray:
        self.floor.update(positions, signals)
        floor = self.floor.history[-1]

        self.kalman_filter.predict()
        detected_position = calculate_activation_cluster_means(floor, self.floor.config.idle_field_value)

        if len(detected_position) == 0:
            self.frames_without_position += 1
            return self.kalman_filter.x / PATCH_SIZE

        if self.frames_without_position > self.filter_reset_threshold:
            self.kalman_filter.reset()

        self.kalman_filter.update(detected_position)
        self.frames_without_position = 0

        return self.kalman_filter.x / PATCH_SIZE
