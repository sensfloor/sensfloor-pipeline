from pathlib import Path

from matplotlib import pyplot as plt

from data_loading.roi_floor import RoIFloorConfig
from definitions import TRAIN_DATA_PATH
from training.dataset.lstm_dataset import LSTMDataset
from training.dataset.dataset_utils import DatasetConfig, get_sequences
from training.dataset.sensfloor_dataset import SensfloorPosesDataset

if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({
    'A': [5, 7, 3],
    'B': [10, 20, 30],
    'C': ['X', 'Y', 'X']
    })

    results = []
    for row in df.itertuples(index=False):
        results.append(row.A * row.B if row.C == 'X' else row.A + row.B)

    df['Result'] = results
    print(df)

    def load_single_dataset(data_path: Path, config: DatasetConfig, return_detailed=False) -> LSTMDataset:
        poses_df = pd.read_csv(data_path / "video_poses.csv")
        readout_df = pd.read_csv(data_path / "sensfloor_readout.csv")
        return LSTMDataset(
            poses_df=poses_df,
            sensfloor_readout_df=readout_df,
            config=config,
            return_detailed=return_detailed,
        )
    
    config=DatasetConfig(floor_config=RoIFloorConfig(x_size=4,y_size=6,history_maxlen=25,roi_size=3))
    folders = [folder for folder in TRAIN_DATA_PATH.iterdir() if folder.is_dir()]

    sequences_dict = {}
    for folder in folders:
        poses_df = pd.read_csv(folder / "video_poses.csv")
        readout_df = pd.read_csv(folder / "sensfloor_readout.csv")
        sequences = get_sequences(poses=poses_df, sensfloor_readout=readout_df)
        print(f"sequences for {folder} {sequences}")
        sequences_dict[folder.name] = sequences

    fig, ax = plt.subplots()
    ax.boxplot(sequences_dict.values())
    ax.set_xticklabels(sequences_dict.keys())
    plt.show()


