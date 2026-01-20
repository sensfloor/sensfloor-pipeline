import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data_collection.message_validator import PositionValidator
from data_collection.messages_provider import CSVMessagesProvider, MessagesProvider, SerialMessagesProvider
from inference.model_loading import load_data_transformations, load_floor, load_model, load_pose_landmark_mapping
from inference.websocket import Websocket
from training.configs import ModelType


def get_message_provider(mock_file: Path | None, serial_port: str | None) -> MessagesProvider:
    if mock_file is None and serial_port is not None:
        print(f"Use serial messages provider with port {serial_port}")
        message_validator = PositionValidator(min_x=1, max_x=6, min_y=1, max_y=4)
        return SerialMessagesProvider(serial_port, message_validator=message_validator)

    if mock_file is not None:
        print(f"Use csv messages provider with file {mock_file}")
        return CSVMessagesProvider(mock_file)

    error_message = "Script requires either serial port or path to csv mock file"
    raise ValueError(error_message)


@torch.no_grad()
def main(fps: int, model_folder: Path, mock_file: Path | None, serial_port: str | None) -> None:
    model, device, model_type = load_model(model_folder)
    transform_data = load_data_transformations(model_folder)
    floor, floor_config = load_floor(model_folder)
    pose_landmark_mapping = load_pose_landmark_mapping(model_folder)

    print(f"Run model({model_type.name}) on {device}")
    model.eval()

    messages_provider = get_message_provider(mock_file, serial_port)
    with Websocket() as websocket, messages_provider:
        frame_interval_length = 1.0 / fps

        h_c = None # used to store history if model is LSTM
        while True:
            frame_start_time = time.perf_counter()

            messages = messages_provider.get_messages()
            positions = []
            signals = []
            for message_dict in messages:
                positions.append([message_dict["x"], message_dict["y"]])
                signals.append([message_dict[f"{i}"] for i in range(8)])

            # IMPORTANT: Floor expects positions starting from 0, sensfloor starts from 1 -> Subtract 1
            floor.update(np.array(positions) - 1, np.array(signals))

            roi = floor.get_roi()

            if roi is not None:
                x = torch.Tensor(roi.history).unsqueeze(0)
                x = x.to(device)
                x = transform_data(x)
                
                # TODO add logic to reset h_c if there were e.g. 15 frames without signal
                if model_type in [ModelType.CNN_LSTM, ModelType.CNN_LSTM_EFFICIENT]:
                    outputs, (h_c) = model(x, h_c)
                else:
                    outputs = model(x)

                joints: list = outputs.reshape(-1, 3).cpu().tolist()

                message = {
                    "x_roi": int(roi.x),
                    "y_roi": int(roi.y),
                    "roi_size": int(floor_config.roi_size),
                    "joints": [
                        {"joint": pose_landmark_mapping[i].name, "x": joint[0], "y": joint[1], "z": joint[2]}
                        for i, joint in enumerate(joints)
                    ],
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
