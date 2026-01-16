import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data_collection.messages_provider import CSVMessagesProvider, SerialMessagesProvider
from data_collection.messages_provider.base_messages_provider import MessagesProvider
from data_loading.roi_floor import RoIFloor, RoIFloorConfig
from inference.websocket import Websocket
from training.pose_estimation_model import RegressionModel
from training.utils import get_device


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
def main(fps: int, model_path: Path, mock_file: Path | None, serial_port: str | None) -> None:
    history_maxlen = 25
    roi_size = 3
    floor_config = RoIFloorConfig(x_size=6, y_size=4, history_maxlen=history_maxlen, roi_size=roi_size)

    device = get_device()
    model = RegressionModel(
        roi_shape=(12, 12),
        landmarks_out=17,
        history_len=history_maxlen,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    floor = RoIFloor(floor_config)
    messages_provider = get_message_provider(mock_file, serial_port)
    with Websocket() as websocket, messages_provider:
        frame_interval_length = 1.0 / fps

        messages_provider.start()
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
                outputs = model(x)
                joints: list = outputs.reshape(-1, 3).cpu().tolist()

                message = {
                    "x_roi": int(roi.x),
                    "y_roi": int(roi.y),
                    "roi_size": int(floor_config.roi_size),
                    "joints": [{"joint": "???", "x": joint[0], "y": joint[1], "z": joint[2]} for joint in joints],
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
        help="Path to model weights file",
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
    main(fps=1, model_path=args.model, mock_file=args.mock_file, serial_port=args.serial_port)
