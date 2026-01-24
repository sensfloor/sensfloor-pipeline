from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from data_loading.pose_landmark import PoseLandmark
from definitions import ROOT_PATH
from training.link_loss import calculate_linkloss
from training.trainer.base_trainer import BaseTrainer


class SensfloorTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        link_min: np.ndarray,
        link_max: np.ndarray,
        pose_to_model_dict: dict[PoseLandmark, int],
        landmarks_out: int,
        kept_links: list[tuple[PoseLandmark, PoseLandmark]],
        scheduler: LRScheduler | None = None,
        use_early_stopping: bool = True,
        patience: int = 10,
        results_path: Path = ROOT_PATH,
        best_model_name: str = "best_model.pth",
        amplify_link_loss: float = 10,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            use_early_stopping=use_early_stopping,
            patience=patience,
            best_model_name=best_model_name,
            results_path=results_path,
        )

        self.amplify_link_loss = amplify_link_loss
        self.pose_to_model_dict = pose_to_model_dict
        self.k_min, self.k_max = link_min, link_max
        self.landmarks_out = landmarks_out
        self.kept_links = kept_links

    @staticmethod
    def calculate_mean_accuracy(outputs, labels, landmarks_out: int, threshold=0.1):
        joint_accuracy = SensfloorTrainer.calculate_joint_accuracies(outputs, labels, landmarks_out, threshold)
        accuracy = joint_accuracy.mean()  # average over all B × landmarks_out
        return accuracy.item() * 100

    @staticmethod
    def calculate_joint_accuracies(outputs, labels, landmarks_out: int, threshold=0.1):
        coords = outputs.view(-1, landmarks_out, 3)  # [B, landmarks_out, 3]
        reshaped_labels = labels.view(-1, landmarks_out, 3)  # [B, landmarks_out, 3]
        dist = torch.linalg.vector_norm(coords - reshaped_labels, dim=2)  # [B, landmarks_out]
        correct = dist < threshold  # values: [True, False, ...]
        return correct.float()  # values: [1, 0, ...]

    def forward_pass(self, inputs: torch.Tensor):
        return self.model(inputs)

    def calculate_loss(self, outputs, labels) -> torch.Tensor:
        mse_loss = nn.MSELoss(reduction="mean")(outputs, labels)
        link_loss = (
            calculate_linkloss(outputs, self.k_min, self.k_max, self.pose_to_model_dict, links=self.kept_links)
            * self.amplify_link_loss
        )  # Paper amplifies link loss by 10
        loss = mse_loss + link_loss
        return loss

    def calculate_accuracy(self, outputs, labels, threshold=0.1):
        return self.calculate_mean_accuracy(outputs, labels, self.landmarks_out, threshold)


def get_test_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader, device, landmarks_out: int):
    model.eval()
    model.to(device)
    total_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            accuracy = SensfloorTrainer.calculate_mean_accuracy(outputs, labels, landmarks_out)
            total_accuracy += accuracy

    return total_accuracy / len(test_loader)
