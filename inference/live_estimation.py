import time
from pathlib import Path

from data_collection.messages_provider import CSVMessagesProvider
from data_loading.roi_floor import RoIFloor, RoIFloorConfig


def main(fps: int) -> None:
    history_maxlen = 10
    roi_size = 3
    floor_config = RoIFloorConfig(x_size=6, y_size=4, history_maxlen=history_maxlen, roi_size=roi_size)

    recording_start_ns = time.perf_counter()
    # model = RegressionModel((3, 3), 10, 17)  # TODO: Adjust arguments
    messages_provider = CSVMessagesProvider(Path("data/2025-12-02_12-24-03/sensfloor_readout.csv"))
    floor = RoIFloor(floor_config)

    try:
        frame_interval_length = 1.0 / fps

        messages_provider.start()
        while True:
            frame_start_time = time.perf_counter()

            messages = messages_provider.get_messages()
            print(messages)
            print("-----------------------")
            positions = []
            signals = []
            for message_dict in messages:
                positions.append([message_dict["x"], message_dict["y"]])
                signals.append([message_dict[f"{i}"] for i in range(8)])

            # floor.update(np.array(positions), np.array(signals))

            # roi = floor.get_roi()

            # outputs = model(torch.Tensor(roi.history))

            frame_duration = time.perf_counter() - frame_start_time
            sleep_time = frame_interval_length - frame_duration

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(
                    f"Processing took longer than {frame_interval_length:.4f}s "
                    f"(took {frame_duration:.4f}s). Video is not {fps} FPS!",
                )

    finally:
        recording_time = time.perf_counter() - recording_start_ns
        print(f"Recorded for {recording_time}s")
        messages_provider.stop()


if __name__ == "__main__":
    main(fps=1)
