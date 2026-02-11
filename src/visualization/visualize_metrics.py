import argparse
from pathlib import Path


from src.training.trainer.sensfloor_trainer import MJPE_PREFIX, get_test_metrics
from src.training.training_pipeline import get_training_setup

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    output_image = "mjpe_metrics_boxplot.png"
    plt.savefig(output_image)
    print(f"Boxplot saved to {output_image}")
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

def load_and_plot_mjpe_professional(file_path: Path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update(
    {
        "font.family": "Courier New",
        "font.size": 15,
    },
)
    
    df = pd.read_csv(file_path)
    
    mjpe_columns = [col for col in df.columns if 'mjpe' in col.lower()]
    
    if not mjpe_columns:
        print(f"No columns found.")
        return df.to_dict(orient='list')

    plot_df = df[mjpe_columns].copy()
    
    # 1. Clean labels and determine colors
    clean_labels = {}
    column_colors = [] # This will hold the color for each specific column
    
    for col in mjpe_columns:
        # Determine category for coloring
        category = get_landmark_category(col)
        column_colors.append(CATEGORY_COLORS[category])
        
        # Clean the name
        # Assuming TEST_PREFIX and MJPE_PREFIX might vary, we just clean common patterns
        new_name = col.lower().replace('test_', '').replace('mjpe_', '').replace('_', ' ').title()
        if new_name.strip() == "Mean": new_name = "Overall Mean"
        clean_labels[col] = new_name
        
    plot_df = plot_df.rename(columns=clean_labels)
    
    plt.figure(figsize=(14, 7), dpi=300)

    # 2. Create the Boxplot with Custom Palette
    # Note: When plotting "wide" data (whole dataframe), we can pass a list of colors 
    # to 'palette' that matches the number of columns.
    ax = sns.boxplot(
        data=plot_df, 
        linewidth=1.2,
        fliersize=2,
        palette=column_colors,  # Pass our specific list of colors here
        width=0.6,
        saturation=0.9
    )

    # 3. Formatting
    #plt.title("Distribution of MJPE", fontsize=16, weight='bold', pad=20)
    plt.ylabel(r"Mean Joint Position Error ($10^{-5}$)", fontsize=12)
    plt.xlabel("") 
    
    plt.xticks(rotation=45, ha='right')
    
    ax.yaxis.grid(True, linestyle='--', which='major', color='black', alpha=0.25)
    ax.xaxis.grid(False)
    sns.despine(trim=True, left=True)

    # 4. Add a Custom Legend
    # Since boxplot doesn't generate a legend for 'palette' lists automatically, we make one
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=CATEGORY_COLORS['Body'], lw=4, label='Body (Head/Face)'),
        Line2D([0], [0], color=CATEGORY_COLORS['Arms'], lw=4, label='Arms'),
        Line2D([0], [0], color=CATEGORY_COLORS['Legs'], lw=4, label='Legs')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Region")

    plt.tight_layout()
    
    output_image = "mjpe_metrics_publication_colored.svg"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.savefig(output_image, format="svg", bbox_inches="tight")
    print(f"Colored boxplot saved to {output_image}")
    plt.close()
    
    return df.to_dict(orient='list')    