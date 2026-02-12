from pathlib import Path
import pandas as pd
from src.data_loading.pose_landmark import PoseLandmark
from src.definitions import TRAIN_DATA_PATH, VIDEO_POSES_FILENAME
import matplotlib.pyplot as plt

def box_plot_all_median():

    all_means = []
    
    for directory in TRAIN_DATA_PATH.iterdir():
        if not directory.is_dir(): continue
        
        poses_path = directory / VIDEO_POSES_FILENAME
        

        poses = pd.read_csv(poses_path)

        rename_columns = {}
        for landmark in PoseLandmark:
            column_name = f"y{landmark.value}"
            if column_name in poses.columns:
                rename_columns[column_name] = landmark.name

        poses = poses.rename(columns=rename_columns)
        landmark_columns = list(rename_columns.values())
        
        # Calculate mean for this specific video
        poses[landmark_columns] = poses[landmark_columns].multiply(100)
        means = poses[landmark_columns].var()
        
        all_means.append(means)

    df_means = pd.DataFrame(all_means)

    plt.figure(figsize=(15, 8)) 
    df_means.boxplot(rot=90) 

    plt.title("Mean Y-Coordinate Distribution per Landmark (Across All Videos)")
    plt.ylabel("Y Coordinate (scaled)")
    plt.tight_layout() # Prevents labels from being cut off
    plt.show()

def boxplot_each_recording():
    data_path = Path("data/train/2025-12-16_12-20-54-line-subject1")

    for directory in TRAIN_DATA_PATH.iterdir():
        poses_path = directory / VIDEO_POSES_FILENAME
        poses = pd.read_csv(poses_path)

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


if __name__ == "__main__":
    boxplot_each_recording()
    box_plot_all_median()