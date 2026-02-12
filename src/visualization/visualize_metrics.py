from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.data_loading.pose_landmark import PoseLandmark
from src.definitions import TEST_METRICS_FILENAME
from src.training.trainer.sensfloor_trainer import MJPE_PREFIX, TEST_PREFIX


LANDMARK_GROUPS = {
    # Body/Head
    'Body': [
        PoseLandmark.LEFT_SHOULDER.name, PoseLandmark.RIGHT_SHOULDER.name,
        PoseLandmark.LEFT_HIP.name, PoseLandmark.RIGHT_HIP.name,
        PoseLandmark.NOSE.name, PoseLandmark.LEFT_EYE_INNER.name, PoseLandmark.LEFT_EYE.name, PoseLandmark.LEFT_EYE_OUTER.name,
        PoseLandmark.RIGHT_EYE_INNER.name, PoseLandmark.RIGHT_EYE.name, PoseLandmark.RIGHT_EYE_OUTER.name,
        PoseLandmark.LEFT_EAR.name, PoseLandmark.RIGHT_EAR.name, PoseLandmark.MOUTH_LEFT.name, PoseLandmark.MOUTH_RIGHT.name
    ],
    # Arms
    'Arms': [
        PoseLandmark.LEFT_ELBOW.name, PoseLandmark.RIGHT_ELBOW.name,
        PoseLandmark.LEFT_WRIST.name, PoseLandmark.RIGHT_WRIST.name, PoseLandmark.LEFT_PINKY.name, PoseLandmark.RIGHT_PINKY.name,
        PoseLandmark.LEFT_INDEX.name, PoseLandmark.RIGHT_INDEX.name, PoseLandmark.LEFT_THUMB.name, PoseLandmark.RIGHT_THUMB.name
    ],
    # Legs
    'Legs': [
        PoseLandmark.LEFT_KNEE.name, PoseLandmark.RIGHT_KNEE.name,
        PoseLandmark.LEFT_ANKLE.name, PoseLandmark.RIGHT_ANKLE.name, PoseLandmark.LEFT_HEEL.name, PoseLandmark.RIGHT_HEEL.name,
        PoseLandmark.LEFT_FOOT_INDEX.name, PoseLandmark.RIGHT_FOOT_INDEX.name
    ]
}

# Define distinct colors for publication (Colorblind safe)
CATEGORY_COLORS = {
    'Body': '#95a5a6',  # Grey/Neutral 
    'Arms': '#f45f74',  # Burnt Orange 
    'Legs': '#00b0be'   # Emerald Green
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
    metrics_df = pd.read_csv(csv_file_path)
    
    # 1. Filter Columns
    mjpe_columns = [col for col in metrics_df.columns if MJPE_PREFIX in col.lower()]
    
    if not mjpe_columns:
        print(f"No columns found.")
        return metrics_df.to_dict(orient='list')

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

    mjpe_columns.sort(key=sort_key)
    mjpe_columns.pop(0)
    metrics_df = metrics_df[mjpe_columns].multiply(100) # Make numbers in cm

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
        plot_data.append(metrics_df[col].dropna().values)
    
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
    ax.set_ylabel(r"Mean Joint Position Error (cm)")
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

    plt.tight_layout(pad=0.0)

    plt.margins(0,0)

    
    output_image = model_path / "test_metrics_boxplot.svg"
    plt.savefig(output_image, format="svg", bbox_inches="tight", pad_inches = 0)
    print(f"Compact boxplot saved to {output_image}")
    plt.close()
    
    return metrics_df.to_dict(orient='list')