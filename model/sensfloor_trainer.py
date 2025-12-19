from typing import Literal

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from data_loading.pose_landmark import PoseLandmark
from model.base_trainer import BaseTrainer
from model.link_loss import calculate_linkloss


class SensfloorTrainer(BaseTrainer):
    def __init__(self, model: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 link_min: np.ndarray,
                 link_max: np.ndarray,
                 pose_to_model_dict: dict[PoseLandmark, int],
                 scheduler: LRScheduler | None = None,
                 use_early_stopping: bool = True,
                 patience: int = 10,
                 best_model_name: str = "best_model.pth",
                 amplify_link_loss: float = 10,
                 loss_reduction: Literal["mean", "sum"] = "mean",
                 ):
        super().__init__(model=model, optimizer=optimizer, device=device, scheduler=scheduler,
                         use_early_stopping=use_early_stopping, patience=patience, best_model_name=best_model_name)

        self.amplify_link_loss = amplify_link_loss
        self.loss_reduction = loss_reduction
        self.pose_to_model_dict = pose_to_model_dict
        self.k_min, self.k_max = link_min, link_max

    @staticmethod
    def calculate_accuracy_2(outputs, labels, threshold=0.1):
        coords = outputs.view(-1, 17, 3)  # [B, 17, 3] #TODO: parametrize landmarks_out
        reshaped_labels = labels.view(-1, 17, 3)  # [B, 17, 3] #TODO: parametrize landmarks_out
        dist = torch.linalg.vector_norm(coords - reshaped_labels, dim=2)  # [B, 17]
        correct = (dist < threshold)
        accuracy = correct.float().mean()  # average over all B × 17
        return accuracy.item() * 100

    def forward_pass(self, inputs: torch.Tensor):
        return self.model(inputs)

    def calculate_loss(self, outputs, labels) -> torch.Tensor:
        mse_loss = nn.MSELoss(reduction=self.loss_reduction)(outputs, labels)
        link_loss = calculate_linkloss(outputs, self.k_min,
                                       self.k_max,
                                       self.pose_to_model_dict) * self.amplify_link_loss  # Paper amplifies link loss by 10
        loss = mse_loss + link_loss
        return loss

    def calculate_accuracy(self, outputs, labels, threshold=0.1):
        return self.calculate_accuracy_2(outputs, labels, threshold)

def get_test_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader):

    total_accuracy = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        accuracy = SensfloorTrainer.calculate_accuracy_2(outputs, labels)
        total_accuracy += accuracy

    return total_accuracy / len(test_loader)
