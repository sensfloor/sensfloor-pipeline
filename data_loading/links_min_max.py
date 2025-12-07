import numpy as np

from data_loading.pose_landmark import LINKS
import pandas as pd

def compute_kmin_kmax(csv_path: str,
                      links=None,
                      lower_percentile: float = 3.0,
                      upper_percentile: float = 97.0):
    if links is None:
        links = LINKS

    df = pd.read_csv(csv_path)

    all_link_lengths = []

    for (a, b) in links:
        xa = df[f"x{a}"].to_numpy()
        ya = df[f"y{a}"].to_numpy()
        za = df[f"z{a}"].to_numpy()

        xb = df[f"x{b}"].to_numpy()
        yb = df[f"y{b}"].to_numpy()
        zb = df[f"z{b}"].to_numpy()

        # calculate all pair of distances between joint a and joint b for all frames
        dx = xa - xb
        dy = ya - yb
        dz = za - zb
        d = np.sqrt(dx * dx + dy * dy + dz * dz)

        all_link_lengths.append(d)

    # calculate k_min and k_max for each joint links
    k_min_list = []
    k_max_list = []

    for d in all_link_lengths:
        k_min_list.append(np.percentile(d, lower_percentile))
        k_max_list.append(np.percentile(d, upper_percentile))

    k_min = np.array(k_min_list)
    k_max = np.array(k_max_list)

    return k_min, k_max