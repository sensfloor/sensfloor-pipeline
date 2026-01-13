import time

import numpy as np
import torch

from data_collection.sensfloor_signals_and_video_collection import (
    FPS,
    read_message_from_queue,
    start_collect_messages_thread,
    stop_collect_messages_thread,
)
from data_loading.roi_floor import RoIFloor, RoIFloorConfig
from model.pose_estimation_model import RegressionModel


def main() -> None:
    history_maxlen = 10
    roi_size = 3
    floor_config = RoIFloorConfig(x_size=6, y_size=4, history_maxlen=history_maxlen, roi_size=roi_size)

    recording_start_ns = time.perf_counter()
    model = RegressionModel((3, 3), 10, 17)  # TODO: Adjust arguments
    floor = RoIFloor(floor_config)

    try:
        frame_number = 0
        frame_interval_length = 1.0 / FPS

        while True:
            frame_start_time = time.perf_counter()

            # Collect all messages for current frame from queue
            timestamp_ns = int(time.perf_counter_ns() - recording_start_ns)

            positions = []
            signals = []
            for message_dict in read_message_from_queue(
                collect_messages_handle.message_queue,
                frame_number,
                timestamp_ns,
            ):
                positions.append([message_dict["x"], message_dict["y"]])
                signals.append([message_dict[f"{i}"] for i in range(8)])

            floor.update(np.array(positions), np.array(signals))

            roi = floor.get_roi()

            if roi is None:
                raise RuntimeError

            outputs = model(torch.Tensor(roi.history))

            frame_duration = time.perf_counter() - frame_start_time
            sleep_time = frame_interval_length - frame_duration

            frame_number += 1

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(
                    f"Processing took longer than {frame_interval_length:.4f}s "
                    f"(took {frame_duration:.4f}s). Video is not {FPS} FPS!",
                )

    finally:
        recording_time = time.perf_counter() - recording_start_ns
        print(f"Recorded for {recording_time}s")
        stop_collect_messages_thread(collect_messages_handle)


if __name__ == "__main__":
    # Start message collection thread
    collect_messages_handle = start_collect_messages_thread()

    try:
        while True:
            # 1. Read messages
            pass
            # 2. Update floor representation
            # 3. Convert messages to tensors
            # 4. Predict next pose
            # 5. Output next pose
    finally:
        stop_collect_messages_thread(collect_messages_handle)
