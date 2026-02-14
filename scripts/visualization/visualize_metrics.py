import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # Needed for NaN
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from src.data_loading.pose_landmark import PoseLandmark
from src.definitions import TEST_METRICS_FILENAME
from src.training.trainer.sensfloor_trainer import MJPE_PREFIX, TEST_PREFIX

LANDMARK_GROUPS = {
    "Body": [
        PoseLandmark.NOSE.name,
        PoseLandmark.LEFT_EYE_INNER.name,
        PoseLandmark.LEFT_EYE.name,
        PoseLandmark.LEFT_EYE_OUTER.name,
        PoseLandmark.RIGHT_EYE_INNER.name,
        PoseLandmark.RIGHT_EYE.name,
        PoseLandmark.RIGHT_EYE_OUTER.name,
        PoseLandmark.LEFT_EAR.name,
        PoseLandmark.RIGHT_EAR.name,
        PoseLandmark.MOUTH_LEFT.name,
        PoseLandmark.MOUTH_RIGHT.name,
        PoseLandmark.LEFT_SHOULDER.name,
        PoseLandmark.RIGHT_SHOULDER.name,
        PoseLandmark.LEFT_HIP.name,
        PoseLandmark.RIGHT_HIP.name,
    ],
    "Arms": [
        PoseLandmark.LEFT_ELBOW.name,
        PoseLandmark.RIGHT_ELBOW.name,
        PoseLandmark.LEFT_WRIST.name,
        PoseLandmark.RIGHT_WRIST.name,
        PoseLandmark.LEFT_PINKY.name,
        PoseLandmark.RIGHT_PINKY.name,
        PoseLandmark.LEFT_INDEX.name,
        PoseLandmark.RIGHT_INDEX.name,
        PoseLandmark.LEFT_THUMB.name,
        PoseLandmark.RIGHT_THUMB.name,
    ],
    "Legs": [
        PoseLandmark.LEFT_KNEE.name,
        PoseLandmark.RIGHT_KNEE.name,
        PoseLandmark.LEFT_ANKLE.name,
        PoseLandmark.RIGHT_ANKLE.name,
        PoseLandmark.LEFT_HEEL.name,
        PoseLandmark.RIGHT_HEEL.name,
        PoseLandmark.LEFT_FOOT_INDEX.name,
        PoseLandmark.RIGHT_FOOT_INDEX.name,
    ],
}

CATEGORY_COLORS = {
    "Body": "#95a5a6",
    "Arms": "#f45f74",
    "Legs": "#00b0be",
}


def get_landmark_category(column_name):
    upper_name = column_name.upper()
    for category, landmarks in LANDMARK_GROUPS.items():
        for landmark in landmarks:
            if landmark in upper_name:
                return category
    return "Body"


def clean_label(col_name):
    clean = col_name.lower().replace(TEST_PREFIX, "").replace(MJPE_PREFIX, "").replace("_", " ").title()
    clean = clean.replace("Left", "L.").replace("Right", "R.")
    if clean.strip() == "Mean":
        return "Overall"
    return clean


def create_mjpe_boxenplot(model_path: Path):
    plt.rcParams.update(
        {
            "font.family": "Courier New",
            "font.size": 16,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.color": "black",
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
        },
    )

    csv_file_path = model_path / TEST_METRICS_FILENAME
    metrics_df = pd.read_csv(csv_file_path)

    # 1. Filter Columns
    mjpe_columns = [col for col in metrics_df.columns if MJPE_PREFIX in col.lower()]
    if not mjpe_columns:
        print("No columns found.")
        return metrics_df.to_dict(orient="list")

    # 2. Strict Sort Logic (Ensures L/R are always adjacent)
    def rigorous_sort_key(col_name):
        if "mean" in col_name.lower():
            return (-1, -1)
        upper = col_name.upper()
        # Search for exact index in the definition lists
        for cat_idx, (cat, landmarks) in enumerate(LANDMARK_GROUPS.items()):
            for lm_idx, lm in enumerate(landmarks):
                if lm in upper:
                    return (cat_idx, lm_idx)
        return (99, 99)

    mjpe_columns.sort(key=rigorous_sort_key)
    if "mean" in mjpe_columns[0].lower():
        mjpe_columns.pop(0)

    # 3. Construct Data with Spacers (The "Gap" Logic)
    df_list = []
    palette_map = {}

    # We will build a specific order for the X-axis
    x_axis_order = []

    for i, col in enumerate(mjpe_columns):
        clean = clean_label(col)
        cat = get_landmark_category(col)

        # Add real data
        df_list.append(
            pd.DataFrame(
                {
                    "Joint": clean,
                    "Error": metrics_df[col] * 100,
                },
            ),
        )
        palette_map[clean] = CATEGORY_COLORS[cat]
        x_axis_order.append(clean)

        if i < len(mjpe_columns) - 1:
            next_col = mjpe_columns[i + 1]
            next_clean = clean_label(next_col)

            # Extract root name (e.g. "L. Shoulder" -> "Shoulder")
            curr_root = clean.replace("L.", "").replace("R.", "").strip()
            next_root = next_clean.replace("L.", "").replace("R.", "").strip()

            # If roots are different, joints are different -> Insert Spacer
            if curr_root != next_root:
                spacer_name = f"__spacer_{i}__"  # Unique ID for spacer

                # Add dummy row with NaN error (Seaborn draws nothing for NaN)
                df_list.append(
                    pd.DataFrame(
                        {
                            "Joint": spacer_name,
                            "Error": [np.nan] * len(metrics_df),
                        },
                    ),
                )

                palette_map[spacer_name] = (0, 0, 0, 0)  # Transparent color
                x_axis_order.append(spacer_name)

    # Combine into one Long-Form DataFrame
    melted_df = pd.concat(df_list, ignore_index=True)

    _, ax = plt.subplots(figsize=(4, 4), dpi=300)

    sns.boxenplot(
        data=melted_df,
        x="Joint",
        y="Error",
        order=x_axis_order,  # Force the specific order containing spacers
        hue="Joint",
        legend=False,
        palette=palette_map,
        ax=ax,
        width=0.6,
        linewidth=0.5,
        k_depth="trustworthy",
        whis=(5, 95),
        showfliers=True,
        flier_kws=dict(marker="o", s=5, alpha=0.4, edgecolor="black", linewidth=0.5),
    )

    # Axis formatting
    ax.set_ylabel("Mean Joint Position Error (cm)")
    ax.set_xlabel("")

    # Custom X-Ticks (Hide the Spacer labels)
    new_labels = []
    for lbl in x_axis_order:
        if "__spacer_" in lbl:
            new_labels.append("")
        else:
            new_labels.append(lbl)

    ax.set_xticks(range(len(x_axis_order)))  # Set ticks at every position
    ax.set_xticklabels(new_labels, rotation=90, ha="center")

    # Hide ticks (small lines) for spacers if desired, or keep them as small dots
    ax.xaxis.grid(False)
    sns.despine(top=True, right=True, left=True, ax=ax)

    # 5. Legend
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["Body"], edgecolor="black", label="Body"),
        Patch(facecolor=CATEGORY_COLORS["Arms"], edgecolor="black", label="Arms"),
        Patch(facecolor=CATEGORY_COLORS["Legs"], edgecolor="black", label="Legs"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)

    plt.tight_layout(pad=0.0)
    plt.margins(0, 0)

    output_image = model_path / "test_metrics_boxenplot_grouped.svg"
    plt.savefig(output_image, format="svg", bbox_inches="tight", pad_inches=0)
    print(f"Grouped boxenplot saved to {output_image}")
    plt.close()

    return metrics_df.to_dict(orient="list")


def create_mjpe_boxplot(model_path: Path):
    plt.rcParams.update(
        {
            "font.family": "Courier New",
            "font.size": 16,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.color": "black",
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
        },
    )

    csv_file_path = model_path / TEST_METRICS_FILENAME
    metrics_df = pd.read_csv(csv_file_path)

    # 1. Filter Columns
    mjpe_columns = [col for col in metrics_df.columns if MJPE_PREFIX in col.lower()]

    if not mjpe_columns:
        print("No columns found.")
        return metrics_df.to_dict(orient="list")

    # 2. SORT COLUMNS LOGIC
    # Priority: Overall Mean -> Body -> Arms -> Legs
    def sort_key(col_name):
        # Always put "Mean" or "Overall" first (index -1)
        if "mean" in col_name.lower():
            return -1

        category = get_landmark_category(col_name)
        # Define the specific order for the rest
        order = ["Body", "Arms", "Legs"]

        try:
            return order.index(category)
        except ValueError:
            return 99  # Put unknown categories at the end

    mjpe_columns.sort(key=sort_key)
    mjpe_columns.pop(0)
    metrics_df = metrics_df[mjpe_columns].multiply(100)  # Make numbers in cm

    # 3. Prepare Data
    plot_data = []
    labels = []
    box_colors = []

    for col in mjpe_columns:
        category = get_landmark_category(col)
        box_colors.append(CATEGORY_COLORS[category])

        # Cleaning labels
        clean = col.lower().replace(TEST_PREFIX, "").replace(MJPE_PREFIX, "").replace("_", " ").title()
        clean = clean.replace("Left", "L.").replace("Right", "R.")
        if clean.strip() == "Mean":
            clean = "Overall"

        labels.append(clean)
        plot_data.append(metrics_df[col].dropna().values)

    # 4. Create Plot
    _, ax = plt.subplots(figsize=(5, 4), dpi=300)

    positions = []
    current_pos = 1.0
    for i in range(len(labels)):
        positions.append(current_pos)

        # Check if the NEXT label is a "Right" version of the same joint
        if i < len(labels) - 1:
            current_label = labels[i].replace("L.", "").strip()
            next_label = labels[i + 1].replace("R.", "").strip()

            if current_label == next_label:
                current_pos += 0.5  # Small spacing for pairs (L/R)
            else:
                current_pos += 0.8  # Larger spacing between different joints

    bplot = ax.boxplot(
        plot_data,
        positions=positions,
        patch_artist=True,
        tick_labels=labels,
        widths=0.4,
        whis=(5, 95),
        showfliers=False,
        flierprops=dict(marker="o", markersize=2, alpha=0.4, markeredgecolor="black"),
    )

    # Color boxes
    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0.75)
        patch.set_alpha(1)

    plt.setp(bplot["medians"], color="black", linewidth=1.0)

    # Axis formatting
    ax.set_ylabel("MPJPE (cm)", fontweight="bold")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha="center")
    plt.ylim(-1, 26)

    # Spines
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # 5. LEGEND ON TOP & ONE LINE
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["Body"], edgecolor="black", label="Body"),
        Patch(facecolor=CATEGORY_COLORS["Arms"], edgecolor="black", label="Arms"),
        Patch(facecolor=CATEGORY_COLORS["Legs"], edgecolor="black", label="Legs"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout(pad=0.0)

    plt.margins(0, 0)

    output_image = model_path / "test_metrics_boxplot.pdf"
    plt.savefig(output_image, format="pdf", bbox_inches="tight", pad_inches=0)
    print(f"Compact boxplot saved to {output_image}")
    plt.close()

    return metrics_df.to_dict(orient="list")


def main(model_path: Path) -> None:
    create_mjpe_boxplot(model_path)


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
