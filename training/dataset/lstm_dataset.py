from dataclasses import replace

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence

from data_loading.roi_floor import create_roi_floor
from training.dataset.dataset_utils import DatasetConfig, remove_noise_messages
from training.dataset.dataset_utils import get_sequences, DetailedSensfloorPosesData
from training.dataset.sensfloor_dataset import SensfloorPosesDataset


class LSTMDataset(SensfloorPosesDataset):
    def __init__(
            self,
            poses_df: pd.DataFrame,
            sensfloor_readout_df: pd.DataFrame,
            config: DatasetConfig,
            return_detailed: bool = False,
    ) -> None:
        super().__init__(poses_df, sensfloor_readout_df, config, return_detailed)

        filter_threshold = 145
        filtered_readout = remove_noise_messages(sensfloor_readout_df,
                                                 filter_threshold)  # TODO: BaseClass also has filtered readout, refactor that one

        self.sequences = get_sequences(
            sensfloor_readout=filtered_readout,
            poses=poses_df,
        )
        # self._analyse_sequences()

    def _analyse_sequences(self) -> None:
        print("frames", [sequence[1] for sequence in self.sequences])
        print("lengths", [sequence[0] for sequence in self.sequences])
        fig, ax = plt.subplots()
        ax.boxplot([sequence[0] for sequence in self.sequences])
        ax.set_xticklabels([f"{len(self.sequences)}"])
        plt.show()

    def __len__(self) -> int:
        return len(self.sequences)

    def get_detailed_data(self, index: int) -> DetailedSensfloorPosesData:
        # TODO: Add transform -> go back 1-10 frames to get more variety of poses
        history_len, frame_number = self.sequences[index]
        updated_floor_config = replace(self.config.floor_config, history_maxlen=history_len)
        floor = create_roi_floor(updated_floor_config, self.sensfloor_readout_df, frame_number)

        return self._get_transformed_data(floor, frame_number)


def add_0_padding(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # Sequences (batch, different_seq_len, features)
    sequences, labels = zip(*batch)

    # (batch, max_sequence_len, features)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_seqs, labels