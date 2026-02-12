from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.definitions import TEST_METRICS_FILENAME
from src.training.trainer.sensfloor_trainer import MJPE_PREFIX, TEST_PREFIX


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
    return 'Body' # fallback

def load_and_plot_mjpe_professional(model_path: Path):
    plt.rcParams.update({
        "font.family": "Courier New",
        "font.size": 16,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": "black",
        "grid.alpha": 0.2,
        "grid.linestyle": "--"
    })
    
    csv_file_path = model_path / TEST_METRICS_FILENAME 
    df = pd.read_csv(csv_file_path)
    
    # 1. Filter Columns
    mjpe_columns = [col for col in df.columns if MJPE_PREFIX in col.lower()]
    
    if not mjpe_columns:
        print(f"No columns found.")
        return df.to_dict(orient='list')

    # 2. SORT COLUMNS LOGIC
    # Priority: Overall Mean -> Body -> Arms -> Legs
    def sort_key(col_name):
        # Always put "Mean" or "Overall" first (index -1)
        if 'mean' in col_name.lower(): 
            return -1
        
        category = get_landmark_category(col_name)
        # Define the specific order for the rest
        order = ['Body', 'Arms', 'Legs']
        
        try:
            return order.index(category)
        except ValueError:
            return 99 # Put unknown categories at the end

    # Apply the sort
    mjpe_columns.sort(key=sort_key)
    mjpe_columns.pop(0)

    # 3. Prepare Data
    plot_data = []
    labels = []
    box_colors = []
    
    for col in mjpe_columns:
        category = get_landmark_category(col)
        box_colors.append(CATEGORY_COLORS[category])
        
        # Cleaning labels
        clean = col.lower().replace(TEST_PREFIX, '').replace(MJPE_PREFIX, '').replace('_', ' ').title()
        clean = clean.replace('Left', 'L.').replace('Right', 'R.')
        if clean.strip() == "Mean": clean = "Overall"
        
        labels.append(clean)
        plot_data.append(df[col].dropna().values)
    
    # 4. Create Plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    bplot = ax.boxplot(plot_data,
                       patch_artist=True,
                       labels=labels,
                       widths=0.4, 
                       flierprops=dict(marker='o', markersize=2, alpha=0.3, markeredgecolor='black'))

    # Color boxes
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0.75)
        patch.set_alpha(1)
    
    plt.setp(bplot['medians'], color='black', linewidth=1.0)
    
    # Axis formatting
    ax.set_ylabel(r"Mean Joint Position Error")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='center') 
    
    # Spines
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) 

    # 5. LEGEND ON TOP & ONE LINE
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS['Body'], edgecolor='black', label='Body'),
        Patch(facecolor=CATEGORY_COLORS['Arms'], edgecolor='black', label='Arms'),
        Patch(facecolor=CATEGORY_COLORS['Legs'], edgecolor='black', label='Legs')
    ]
    
    # bbox_to_anchor=(x, y): (0.5, 1.0) is top-center.
    # loc='lower center' means the bottom of the legend box is at that point.
    # ncol=3 forces the items into a single row.
    ax.legend(handles=legend_elements, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 1.0), 
              ncol=3, 
              frameon=False, 
              fontsize=9)

    plt.tight_layout(pad=0.2)
    
    output_image = model_path / "test_metrics_boxplot.svg"
    plt.savefig(output_image, format="svg", bbox_inches="tight")
    print(f"Compact boxplot saved to {output_image}")
    plt.close()
    
    return df.to_dict(orient='list')