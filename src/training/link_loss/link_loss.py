import torch

from src.data_loading.pose_landmark import PoseLandmark


def calculate_linkloss(pred_keypoints: torch.Tensor, k_min, k_max, pose_to_model_dict, links: list[tuple[PoseLandmark, PoseLandmark]]):
    device = pred_keypoints.device
    dtype = pred_keypoints.dtype
    coords = pred_keypoints.view(-1, len(pose_to_model_dict), 3)  # [B, landmarks_out, 3]

    link_lengths = []

    for (a, b) in links:  # a, b are the indices of the joints from media pipe (0-32)
        if not a in pose_to_model_dict or not b in pose_to_model_dict:
            print(f"Model outputs do not contain joint with index {a} or {b}, skipping")
            continue
        # model outputs might not have all indices based on hyperparameters
        diff = coords[:, pose_to_model_dict[a]] - coords[:, pose_to_model_dict[b]]  # [B, 3]
        length = torch.linalg.vector_norm(diff, dim=1) + 1e-8 # prevent NaN gradients if distance is 0
        link_lengths.append(length)

    link_lengths = torch.stack(link_lengths, dim=1)

    k_min = torch.as_tensor(k_min, device=device, dtype=dtype)  # convert to tensor
    k_max = torch.as_tensor(k_max, device=device, dtype=dtype)

    short_linkloss = torch.clamp(k_min - link_lengths, min=0.0)
    long_linkloss = torch.clamp(link_lengths - k_max, min=0.0)
    link_loss = short_linkloss + long_linkloss

    return link_loss.mean()
