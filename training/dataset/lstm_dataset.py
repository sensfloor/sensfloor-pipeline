import pandas as pd
from matplotlib import pyplot as plt

from data_loading.roi_floor import create_roi_floor, RoIFloorConfig
from training.dataset.dataset_utils import DatasetConfig, get_sequences, DetailedSensfloorPosesData
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
        self.sequences = get_sequences(
            sensfloor_readout=self.filtered_readout,
            poses=poses_df,
        )
        fig, ax = plt.subplots()
        ax.boxplot([self.sequences])
        ax.set_xticklabels([f"{len(self.sequences)}"]) #TODO: Remove boxplotting
        plt.show()

    def __len__(self) -> int:
        return len(self.sequences)

    def get_detailed_data(self, index: int) -> DetailedSensfloorPosesData:
        frame_number, history_len = self.sequences[index]  #TODO: Return and load sequences with correct length
        old = self.config.floor_config
        floor = create_roi_floor(
            RoIFloorConfig(history_maxlen=history_len, x_size=old.x_size, y_size=old.y_size, roi_size=old.roi_size),
            self.sensfloor_readout_df, frame_number)

        return self._get_transformed_data(floor, frame_number)
