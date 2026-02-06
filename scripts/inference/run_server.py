import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.data_collection.messages_provider import CSVMessagesProvider, MessagesProvider, SerialMessagesProvider
from src.data_loading.floor import FloorConfig
from src.inference.pose_predictor import PosePredictor
from src.inference.websocket import Websocket
from src.tracking.kalman_filter import FILTER_OPTIMIZED_ACTIVE_FIELD_MIN_VALUE
from src.tracking.person_tracker import PersonTracker


def get_message_provider(mock_file: Path | None, serial_port: str | None) -> MessagesProvider:
    if mock_file is None and serial_port is not None:
        print(f"Use serial messages provider with port {serial_port}")
        return SerialMessagesProvider(serial_port)

    if mock_file is not None:
        print(f"Use csv messages provider with file {mock_file}")
        return CSVMessagesProvider(mock_file)

    error_message = "Script requires either serial port or path to csv mock file"
    raise ValueError(error_message)


@torch.no_grad()
def main(fps: int, model_folder: Path, mock_file: Path | None, serial_port: str | None) -> None:
    pose_predictor = PosePredictor(model_folder=model_folder, num_calls_cache=5)

    person_tracker_floor_config = FloorConfig(
        x_size=pose_predictor.floor_config.x_size,
        y_size=pose_predictor.floor_config.y_size,
        history_maxlen=1,
        active_field_min_value=FILTER_OPTIMIZED_ACTIVE_FIELD_MIN_VALUE,
        remove_noise=True,
    )
    person_tracker = PersonTracker(fps=fps, filter_reset_threshold=10, floor_config=person_tracker_floor_config)

    messages_provider = get_message_provider(mock_file, serial_port)
    with Websocket() as websocket, messages_provider:
        frame_interval_length = 1.0 / fps

        while True:
            frame_start_time = time.perf_counter()

            # Update floor
            messages = messages_provider.get_messages()
            positions = []
            signals = []
            for message_dict in messages:
                positions.append([message_dict["x"], message_dict["y"]])
                signals.append([message_dict[f"{i}"] for i in range(8)])

            positions = np.array(positions) - 1
            signals = np.array(signals)

            # Predict pose and track person
            pose = pose_predictor.predict(positions, signals)
            position = person_tracker.track(positions, signals)

            # Send message to client
            message = {
                "position_x": position[0],
                "position_y": position[1],
                "pose": pose,
            }
            websocket.send_poses(message)

            frame_duration = time.perf_counter() - frame_start_time
            sleep_time = frame_interval_length - frame_duration

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(
                    f"Processing took longer than {frame_interval_length:.4f}s "
                    f"(took {frame_duration:.4f}s). Video is not {fps} FPS!",
                )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live prediction of poses")

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model folder containing weights and configuration file",
    )

    parser.add_argument(
        "--mock-file",
        type=Path,
        default=None,
        help="Path to csv file containing recorded sensfloor signals to mock live sensor",
    )

    parser.add_argument(
        "--serial-port",
        type=str,
        help="Serial port of sensfloor connector.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(fps=15, model_folder=args.model, mock_file=args.mock_file, serial_port=args.serial_port)
