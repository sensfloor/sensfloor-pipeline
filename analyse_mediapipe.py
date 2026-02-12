from pathlib import Path
import pandas as pd
from src.data_loading.pose_landmark import PoseLandmark
from src.definitions import TRAIN_DATA_PATH, VIDEO_POSES_FILENAME
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # This variable was unused in your loop, so I commented it out
    # data_path = Path("data/train/2025-12-16_12-20-54-line-subject1")

    all_means = []
    
    # We need to make sure we are iterating correctly
    # (Assuming TRAIN_DATA_PATH is defined)
    for directory in TRAIN_DATA_PATH.iterdir():
        if not directory.is_dir(): continue # Skip non-directories
        
        poses_path = directory / VIDEO_POSES_FILENAME
        
        # specific check to ensure file exists before reading
        if not poses_path.exists(): continue 

        poses = pd.read_csv(poses_path)

        rename_columns = {}
        # Assuming PoseLandmark is imported from mediapipe or similar
        for landmark in PoseLandmark:
            column_name = f"y{landmark.value}"
            # Check if column actually exists to avoid KeyErrors
            if column_name in poses.columns:
                rename_columns[column_name] = landmark.name

        poses = poses.rename(columns=rename_columns)
        
        # Filter to only keep the columns we renamed (the landmarks)
        landmark_columns = list(rename_columns.values())
        
        # Calculate mean for this specific video
        poses[landmark_columns] = poses[landmark_columns].multiply(100)
        means = poses[landmark_columns].mean()
        
        all_means.append(means)

    # --- FIX STARTS HERE ---
    
    # 1. Convert the list of Series into a single DataFrame
    # Rows = Different Videos, Columns = Landmarks
    df_means = pd.DataFrame(all_means)

    # 2. Setup the plot
    plt.figure(figsize=(15, 8)) # Make it wide enough to read labels

    # 3. Plot using Pandas directly (it handles labels better)
    # This creates one box per column (Landmark)
    df_means.boxplot(rot=90) 

    plt.title("Mean Y-Coordinate Distribution per Landmark (Across All Videos)")
    plt.ylabel("Y Coordinate (scaled)")
    plt.tight_layout() # Prevents labels from being cut off
    plt.show()