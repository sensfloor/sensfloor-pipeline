import torch

from data_loading.pose_landmark import LINKS


def calculate_linkloss(pred_keypoints: torch.Tensor, k_min, k_max):
    device = pred_keypoints.device
    dtype = pred_keypoints.dtype

    link_lengths = []

    for (a, b) in LINKS:  # a, b are indices of the connected keypoints
        diff = pred_keypoints[:, a] - pred_keypoints[:, b]
        d = torch.linalg.vector_norm(diff, dim=1)
        link_lengths.append(d)

    link_lengths = torch.stack(link_lengths, dim=1)

    k_min = torch.as_tensor(k_min, device=device, dtype=dtype)  # convert to tensor
    k_max = torch.as_tensor(k_max, device=device, dtype=dtype)

    short_linkloss = torch.clamp(k_min - link_lengths,
                                 min=0.0)  # if the link length is less than k_min then return k_min - link_lengths, otherwise 0
    long_linkloss = torch.clamp(link_lengths - k_max,
                                min=0.0)  # if the link length is bigger than k_max then return link_lengths - k_max, otherwise 0
    link_loss = short_linkloss + long_linkloss

    return link_loss.sum()