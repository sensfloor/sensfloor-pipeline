import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from data_loading.pose_landmark import PoseLandmark
from model.base_trainer import BaseTrainer

# All Links between joints
LINKS = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW), # put symmetric links next to each other to compare them
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
    (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX)
]

data_path = './data/2025-12-02_12-08-00/video_poses.csv'


def compute_kmin_kmax(csv_path: str,
                      links: list[tuple[int, int]],
                      lower_percentile: float = 3.0,
                      upper_percentile: float = 97.0):
    df = pd.read_csv(csv_path)

    num_links = len(links)
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


k_min, k_max = compute_kmin_kmax(data_path, LINKS)
print(k_min, k_max)

def loss(logits: torch.Tensor, labels: torch.Tensor):
    """
    logits: [B, 63]
    labels: [B, 63]
    """

    mse_loss = nn.MSELoss(reduction='mean')(logits, labels)
    link_loss = calculate_linkloss(logits, k_min, k_max) / len(LINKS)
    loss = mse_loss + link_loss
    # TODO: Add loss for too large joints / regularization -> So the model doesnt go local minimum setting all points 0
    return loss


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


class SensfloorTrainer(BaseTrainer):
    def __init__(self, model: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 scheduler: LRScheduler | None = None,
                 use_early_stopping: bool = True,
                 patience: int = 10,
                 best_model_name: str = "best_model.pth"):
        super().__init__(model=model, optimizer=optimizer, loss=loss, device=device, scheduler=scheduler,
                         use_early_stopping=use_early_stopping, patience=patience, best_model_name=best_model_name)

    def forward_pass(self, inputs: torch.Tensor):
        return self.model(inputs)

    def calculate_loss(self, outputs, labels) -> torch.Tensor:
        return self.loss(outputs, labels)

    def calculate_accuracy(self, outputs, labels, threshold=0.1):
        coords = outputs.view(-1, 17, 3)  # [B, 17, 3] #TODO: parametrize landmarks_out
        reshaped_labels = labels.view(-1, 17, 3)  # [B, 17, 3] #TODO: parametrize landmarks_out
        dist = torch.linalg.vector_norm(coords - reshaped_labels, dim=2)  # [B, 17]
        correct = (dist < threshold)
        accuracy = correct.float().mean()  # average over all B × 17
        return accuracy * 100
