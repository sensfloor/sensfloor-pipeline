import numpy as np
from scipy.ndimage import center_of_mass, label, sum_labels


def calculate_activation_cluster_means(floor_activations: np.ndarray, idle_field_value: int) -> np.ndarray:
    mask = floor_activations > idle_field_value

    labeled_array, num_features = label(mask)  # type: ignore

    if num_features == 0:
        return np.array([])

    labels = np.arange(1, num_features + 1)
    means = np.array(center_of_mass(mask, labeled_array, labels))

    field_values = sum_labels(floor_activations, labeled_array, labels)
    highest_means = means[np.argsort(field_values)[::-1]][:2]

    if len(highest_means) > 1:
        return highest_means.mean(axis=0).flatten()
    return highest_means.flatten()
