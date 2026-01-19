import time
from pathlib import Path

import pandas as pd

from .base_messages_provider import BaseMessagesProvider


class CSVMessagesProvider(BaseMessagesProvider):
    def __init__(self, csv_path: Path) -> None:
        super().__init__()
        self.csv_path = csv_path

    def _run(self) -> None:
        df = pd.read_csv(self.csv_path)
        df = df.drop(columns=["frame_number"])

        if df.empty:
            return

        start_ts_ns = df.iloc[0]["timestamp"]
        start_real_time_ns = time.perf_counter_ns()

        for _, row in df.iterrows():
            if self._stop_event.is_set():
                break

            original_elapsed_ns = row["timestamp"] - start_ts_ns

            real_elapsed_ns = time.perf_counter_ns() - start_real_time_ns

            wait_time_s = (original_elapsed_ns - real_elapsed_ns) / 1e9
            if wait_time_s > 0:
                time.sleep(wait_time_s)

            message = row.to_dict()

            clean_message = {str(k): int(v) for k, v in message.items()}

            self.messages_queue.put(clean_message)
