import argparse
import csv
from pathlib import Path
from src.definitions import TRAIN_DATA_PATH
from src.training.dataset.load_data import train_val_test_split
import torch
import trackio

from src.data_loading.floor import PATCH_SIZE
from src.data_loading.pose_landmark import PoseLandmark
from src.data_loading.roi_floor import RoIFloorConfig
from src.definitions import HOLD_OUT_DATA_PATH, TRAIN_DATA_PATH
from src.training.configs import (
    CONFIG_FILE_NAME,
    MODELS_FOLDER_PATH,
    PROJECT_NAME,
    PROJECT_GROUP,
    TrainingConfiguration,
)
from src.training.dataset.load_data import train_val_test_split
from src.training.dataset.sensfloor_dataset import DatasetConfig
from src.training.link_loss.links_min_max import get_link_min_max
from src.training.trainer.sensfloor_trainer import MJPE_PREFIX, TEST_PREFIX, TRAIN_PREFIX, SensfloorTrainer, get_test_accuracy, get_test_metrics
from src.training.training_pipeline import get_training_setup
from src.training.utils import get_device, get_kept_links, get_model, set_seed
from src.visualization.create_landmark_predictions import create_predictions



TEST_METRICS_FILENAME = "test_metrics.csv"

def get_and_save_test_metrics(configuration: TrainingConfiguration) -> None:
    model, device, trainer, train_loader, val_loader, test_loader = get_training_setup(configuration)
    test_metrics = get_test_metrics(model, test_loader, device, trainer)
    keys = test_metrics[-1].keys()
    trainer.save_metrics_to_csv(keys, test_metrics, file_name=TEST_METRICS_FILENAME)

import pandas as pd
import matplotlib.pyplot as plt

def load_and_plot_mjpe(file_path: Path):
    df = pd.read_csv(file_path)
    
    mjpe_columns = [col for col in df.columns if MJPE_PREFIX in col]
    
    if not mjpe_columns:
        print(f"No columns containing '{MJPE_PREFIX}' found.", df)
        return df.to_dict(orient='list')

    plt.figure(figsize=(15, 8)) 

    df.boxplot(column=mjpe_columns, rot=90)
    
    plt.title("Distribution of MJPE Metrics per Joint")
    plt.ylabel("Mean Joint Position Error (MJPE)")
    plt.tight_layout()
    
    # Save the plot
    output_image = "mjpe_metrics_boxplot.png"
    plt.savefig(output_image)
    print(f"Boxplot saved to {output_image}")
    plt.close()
    
    # Return the dataframe content as a dictionary
    return df.to_dict(orient='list')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_plot_mjpe_professional(file_path: Path):
    # Set publication-quality style
    # Uses a clean white background with a subtle grid
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    df = pd.read_csv(file_path)
    
    # Filter columns
    mjpe_columns = [col for col in df.columns if 'mjpe' in col.lower()] # Assuming MJPE_PREFIX is just part of the string
    
    if not mjpe_columns:
        print(f"No columns found.")
        return df.to_dict(orient='list')

    # Create a cleaner version of the data for plotting
    plot_df = df[mjpe_columns].copy()
    
    # 1. Clean the labels: Remove prefixes, underscores, and Capitalize
    # Example: 'test_mjpe_LEFT_SHOULDER' -> 'Left Shoulder'
    clean_labels = {}
    for col in mjpe_columns:
        new_name = col.lower().replace(TEST_PREFIX, '').replace(MJPE_PREFIX, '').replace('_', ' ').title()
        # specific fix for 'Mean' if it exists
        if new_name.strip() == "Mean": 
            new_name = "Overall Mean"
        clean_labels[col] = new_name
        
    plot_df = plot_df.rename(columns=clean_labels)
    
    # Initialize the figure with high resolution settings
    plt.figure(figsize=(12, 6), dpi=300)

    # 2. Create the Boxplot
    # We use a distinct palette and reduce the width of the boxes for elegance
    ax = sns.boxplot(
        data=plot_df, 
        linewidth=1.2,       # Thinner lines look sharper
        fliersize=2,         # Smaller outlier points (circles)
        palette="Blues",     # A monochromatic palette is often safer for B&W printing
        width=0.6
    )

    # 3. Formatting
    plt.title("Distribution of MJPE Metrics per Joint", fontsize=16, weight='bold', pad=20)
    
    # Use LaTeX formatting for the Y-axis label if applicable
    plt.ylabel(r"Mean Joint Position Error ($10^{-5}$)", fontsize=12)
    plt.xlabel("") # Often obvious from context, saves space
    
    # Rotate x-labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Refine the Y-axis grid
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    ax.xaxis.grid(False) # Vertical lines are usually unnecessary chart junk
    
    # Remove top and right spines (borders)
    sns.despine(trim=True, left=True)

    plt.tight_layout()
    
    # Save the plot
    output_image = "mjpe_metrics_publication.png"
    # bbox_inches='tight' prevents labels from being cut off
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Professional boxplot saved to {output_image}")
    plt.close()
    
    return df.to_dict(orient='list')
    


def main(model_path: Path):
    config = TrainingConfiguration.load(model_path / CONFIG_FILE_NAME)
    # get_and_save_test_metrics(config)
    load_and_plot_mjpe_professional(model_path / TEST_METRICS_FILENAME)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live prediction of poses")

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model folder containing weights and configuration file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.model)