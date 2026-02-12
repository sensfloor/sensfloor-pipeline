import argparse
from pathlib import Path

from matplotlib.patches import Patch


from src.definitions import TEST_METRICS_FILENAME
from src.training.trainer.sensfloor_trainer import MJPE_PREFIX, get_test_metrics
from src.training.training_pipeline import get_training_setup

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    output_image_path = "mjpe_metrics_boxplot.png"
    plt.savefig(output_image_path)
    print(f"Boxplot saved to {output_image_path}")
    plt.close()
    
    # Return the dataframe content as a dictionary
    return df.to_dict(orient='list')

# Define the grouping logic based on MediaPipe Pose landmarks
LANDMARK_GROUPS = {
    # Body/Head (Indices 0-10)
    'Body': [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
        'LEFT_HIP', 'RIGHT_HIP', 
        'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 
        'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 
        'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT'
    ],
    # Arms (Indices 11-22) including Shoulders
    'Arms': [
        'LEFT_ELBOW', 'RIGHT_ELBOW', 
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 
        'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB'
    ],
    # Legs (Indices 23-32) including Hips
    'Legs': [
        'LEFT_KNEE', 'RIGHT_KNEE', 
        'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 
        'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
    ]
}

# Define distinct colors for publication (Colorblind safe)
CATEGORY_COLORS = {
    'Body': '#95a5a6',  # Grey/Neutral for Head/Torso
    'Arms': '#e67e22',  # Burnt Orange for Upper Limbs
    'Legs': '#2ecc71'   # Emerald Green for Lower Limbs
}

def get_landmark_category(column_name):
    """Finds which category a column belongs to."""
    upper_name = column_name.upper()
    for category, landmarks in LANDMARK_GROUPS.items():
        # Check if any landmark name appears in the column name
        for landmark in landmarks:
            if landmark in upper_name:
                return category
    return 'Body' # Default fallback

def load_and_plot_mjpe_professional(model_path: Path):
    # Set font globally for Matplotlib
    plt.rcParams.update({
        "font.family": "Courier New",
        "font.size": 12, # Slightly smaller base size usually looks better on plots
        "axes.grid": True,
        "grid.color": "black",
        "grid.alpha": 0.25,
        "grid.linestyle": "--"
    })
    
    csv_file_path = model_path / TEST_METRICS_FILENAME 

    df = pd.read_csv(csv_file_path)
    
    mjpe_columns = [col for col in df.columns if MJPE_PREFIX in col.lower()]
    
    if not mjpe_columns:
        print(f"No columns found.")
        return df.to_dict(orient='list')

    plot_data = []
    labels = []
    box_colors = []
    
    for col in mjpe_columns:
        # Determine color
        category = get_landmark_category(col)
        box_colors.append(CATEGORY_COLORS[category])
        
        # Clean label
        new_name = col.lower().replace(TEST_METRICS_FILENAME, '').replace(MJPE_PREFIX, '').replace('_', ' ').title()
        if new_name.strip() == "Mean": new_name = "Overall Mean"
        labels.append(new_name)
        
        # Get data (drop NaNs to avoid plotting errors)
        plot_data.append(df[col].dropna().values)
    
    # Create Figure
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    # Create the Boxplot
    # patch_artist=True is required to fill the boxes with color
    bplot = ax.boxplot(plot_data,
                       patch_artist=True,
                       labels=labels,
                       flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Color the boxes individually
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.2)
        patch.set_alpha(0.9) # Matches your saturation setting

    # Style other elements to be professional (black lines instead of default blue)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bplot[element], color='black', linewidth=1.2)
    
    # Specific styling for median line if you want it distinct
    plt.setp(bplot['medians'], color='black', linewidth=1.5)

    # Formatting
    ax.set_ylabel(r"Mean Joint Position Error ($10^{-5}$)", fontsize=12)
    ax.set_xlabel("")
    
    # Rotate x-labels
    plt.xticks(rotation=90, ha='right')
    
    # Remove vertical grid lines (keep horizontal)
    ax.xaxis.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Optional: cleaner look
    
    # Create Custom Legend using Patches (Squares) instead of Lines
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS['Body'], edgecolor='black', label='Body'),
        Patch(facecolor=CATEGORY_COLORS['Arms'], edgecolor='black', label='Arms'),
        Patch(facecolor=CATEGORY_COLORS['Legs'], edgecolor='black', label='Legs')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Region", frameon=True)

    plt.tight_layout()
    
    # Save output
    output_image = model_path / "test_metrics_boxplot.svg"
    plt.savefig(output_image, format="svg", bbox_inches="tight")
    print(f"Colored boxplot saved to {output_image}")
    plt.close()
    
    return df.to_dict(orient='list')