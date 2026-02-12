from pathlib import Path
import pandas as pd
from src.data_loading.pose_landmark import PoseLandmark
from src.definitions import TRAIN_DATA_PATH, VIDEO_POSES_FILENAME


if __name__ == "__main__":
    data_path = Path("data/train/2025-12-16_12-20-54-line-subject1")

    for directory in TRAIN_DATA_PATH.iterdir():
        poses_path = directory / VIDEO_POSES_FILENAME
        poses = pd.read_csv(poses_path)

        columns = []
        rename_columns = {}

        for landmark in PoseLandmark:
            column_name = f"y{landmark.value}"
            rename_columns[column_name] = landmark.name
            
        poses = poses.rename(columns=rename_columns)
        columns = list(rename_columns.values())

        poses[columns] = poses[columns].multiply(100)
        poses[columns].boxplot()

        import matplotlib.pyplot as plt

        plt.title(directory.name)
        plt.xticks(rotation=90, ha='center') 
        plt.show()