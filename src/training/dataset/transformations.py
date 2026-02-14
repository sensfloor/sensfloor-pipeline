import math

import torch


def normalize_roi(roi: torch.Tensor, idle_floor_value: int, normalize_to_max: bool) -> torch.Tensor:
    normalizes_roi = roi - idle_floor_value
    normalize_to = normalizes_roi.max() if normalize_to_max else idle_floor_value
    return normalizes_roi / normalize_to


def rotate_pose(pose: torch.Tensor, degree: int) -> torch.Tensor:
    num_joints = pose.shape[0] // 3
    reshaped_pose = pose.reshape(num_joints, 3).T
    rad = math.radians(degree)
    # Rotation matrix for rotating around y-axis
    rotation_matrix = torch.tensor(
        [
            [math.cos(rad), 0, math.sin(rad)],
            [0, 1, 0],
            [-math.sin(rad), 0, math.cos(rad)],
        ],
    )
    rotated_pose = rotation_matrix @ reshaped_pose
    return rotated_pose.T.reshape(num_joints * 3)


def rotate_roi(roi: torch.Tensor, degree: int) -> torch.Tensor:
    if degree % 90 != 0:
        message = f"Can only rotate roi in 90° steps (tried rotating by {degree}°)..."
        raise ValueError(message)

    num_90_rotations = degree // 90
    return roi.rot90(k=num_90_rotations, dims=(1, 2))
