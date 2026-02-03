from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from definitions import TRAIN_DATA_PATH, HOLD_OUT_DATA_PATH, READOUT_FILENAME


def visualize_read_outs(data_path: Path, name: str):
    floor_readout = pd.read_csv(data_path / READOUT_FILENAME)
    floor_readout = floor_readout.drop(columns=["timestamp", "frame_number", "group_id", "magic_number"])

    # 2. Calculate the mean of the 8 sub-sensors for each row
    sensor_columns = ['0', '1', '2', '3', '4', '5', '6', '7']
    floor_readout['sensor_val'] = floor_readout[sensor_columns].mean(axis=1)

    # 3. Shift the baseline (CRITICAL STEP)
    # We subtract 127 so that the 'resting' state is 0.
    # Now, simply appearing in the CSV doesn't increase heat unless the value is > 127.
    floor_readout['patch'] = floor_readout['sensor_val'] - 127

    # Clip negative values to 0 just in case there's any noise below 127
    floor_readout['patch'] = floor_readout['patch'].clip(lower=0)

    # 4. Sum over the groups
    # This aggregates the total activity.
    # Frequent occurrences will add up. Rare occurrences will stay low.
    grid_groups = floor_readout.groupby(['x', 'y'])['patch'].sum()

    # 5. Reshape and Plot
    heatmap_data = grid_groups.unstack(level='x').fillna(0)  # Fill missing grid points with 0

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='inferno', origin='lower')
    plt.colorbar(label='Total Accumulated Activity (Intensity * Frequency)')
    plt.title(f"Cumulative Activity Heatmap: {data_path.name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

if __name__ == '__main__':
    training_folders = [directory for directory in TRAIN_DATA_PATH.iterdir() if directory.is_dir()]
    hold_out_folders = [directory for directory in HOLD_OUT_DATA_PATH.iterdir() if directory.is_dir()]
    all_folders = training_folders + hold_out_folders

    for folder in all_folders:
        visualize_read_outs(folder, folder.name)

