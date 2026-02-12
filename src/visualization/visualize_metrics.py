from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # NEW IMPORT
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
    def sort_key(col_name):
        if 'mean' in col_name.lower(): 
            return -1
        category = get_landmark_category(col_name)
        order = ['Body', 'Arms', 'Legs']
        try:
            return order.index(category)
        except ValueError:
            return 99 

    mjpe_columns.sort(key=sort_key)
    mjpe_columns.pop(0) 
    
    subset_df = metrics_df[mjpe_columns].multiply(100)

    # 3. Prepare Data for Seaborn (Long Format & Color Map)
    rename_map = {}
    palette_map = {}
    
    for col in mjpe_columns:
        clean = col.lower().replace(TEST_PREFIX, '').replace(MJPE_PREFIX, '').replace('_', ' ').title()
        clean = clean.replace('Left', 'L.').replace('Right', 'R.')
        if clean.strip() == "Mean": clean = "Overall"
        
        rename_map[col] = clean
        
        cat = get_landmark_category(col)
        palette_map[clean] = CATEGORY_COLORS[cat]

    subset_df = subset_df.rename(columns=rename_map)
    melted_df = subset_df.melt(var_name='Joint', value_name='Error')

    # 4. Create Plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    sns.boxenplot(
        data=melted_df,
        x='Joint',
        y='Error',
        hue='Joint',
        legend=False,
        palette=palette_map,
        ax=ax,
        width=0.6,
        linewidth=0.5,
        k_depth='trustworthy',
        showfliers=True,
        flier_kws=dict(
            marker='o',
            s=5,
            alpha=0.4,
            edgecolor='black',
            linewidth=0.5  
        )
    )
    # Axis formatting
    ax.set_ylabel(r"Mean Joint Position Error (cm)")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='center')
    # Spines
    ax.xaxis.grid(False)
    sns.despine(top=True, right=True, left=True, ax=ax) # Seaborn cleaner equivalent

    # 5. LEGEND
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS['Body'], edgecolor='black', label='Body'),
        Patch(facecolor=CATEGORY_COLORS['Arms'], edgecolor='black', label='Arms'),
        Patch(facecolor=CATEGORY_COLORS['Legs'], edgecolor='black', label='Legs')
    ]
    
    ax.legend(handles=legend_elements,
              loc='lower center', 
              bbox_to_anchor=(0.5, 1.0), 
              ncol=3, 
              frameon=False, 
              fontsize=9)

    plt.tight_layout(pad=0.0)
    plt.margins(0,0)
    
    output_image = model_path / "test_metrics_boxenplot.svg"
    plt.savefig(output_image, format="svg", bbox_inches="tight", pad_inches = 0)
    print(f"Compact boxenplot saved to {output_image}")
    plt.close()
    
    return metrics_df.to_dict(orient='list')