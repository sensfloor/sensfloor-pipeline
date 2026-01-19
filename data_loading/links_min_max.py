from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray

from data_loading.pose_landmark import PoseLandmark
from definitions import DATA_PATH, ROOT_PATH


def get_link_min_max(
    do_compute_link_lengths: bool,
    links: list[tuple[PoseLandmark, PoseLandmark]],
    traing_folders: Path = DATA_PATH,
):
    link_path = ROOT_PATH / "outputs" / "link_lengths"
    link_min_path = link_path / "link_min.npy"
    link_max_path = link_path / "link_max.npy"

    if do_compute_link_lengths:
        link_path.mkdir(parents=True, exist_ok=True)
        link_min, link_max = compute_kmin_kmax(traing_folders, links=links, lower_percentile=3.0, upper_percentile=97.0)
        np.save(link_min_path, link_min)
        np.save(link_max_path, link_max)
    else:
        link_min = np.load(link_min_path)
        link_max = np.load(link_max_path)

    return link_min, link_max


def compute_kmin_kmax(
    train_folder_path: Path,
    links: list[tuple[PoseLandmark, PoseLandmark]],
    lower_percentile: float = 3.0,
    upper_percentile: float = 97.0,
) -> tuple[ndarray, ndarray]:
    collected_link_data = [[] for _ in links]

    folders = [folder for folder in train_folder_path.iterdir() if folder.is_dir()]

    for folder in folders:
        csv_path = folder / "video_poses.csv"
        df = pd.read_csv(csv_path)
        for i, (a, b) in enumerate(links):
            xa = df[f"x{a.value}"].to_numpy()
            ya = df[f"y{a.value}"].to_numpy()
            za = df[f"z{a.value}"].to_numpy()

            xb = df[f"x{a.value}"].to_numpy()
            yb = df[f"y{a.value}"].to_numpy()
            zb = df[f"z{a.value}"].to_numpy()

            # calculate all pair of distances between joint a and joint b for all frames
            dx = xa - xb
            dy = ya - yb
            dz = za - zb
            d = np.sqrt(dx * dx + dy * dy + dz * dz)

            collected_link_data[i].append(d)

    # calculate k_min and k_max for each joint links
    k_min_list = []
    k_max_list = []

    for link_arrays in collected_link_data:
        combined_d = np.concatenate(link_arrays)

        k_min_list.append(np.percentile(combined_d, lower_percentile))
        k_max_list.append(np.percentile(combined_d, upper_percentile))

    k_min = np.array(k_min_list)
    k_max = np.array(k_max_list)

    return k_min, k_max
