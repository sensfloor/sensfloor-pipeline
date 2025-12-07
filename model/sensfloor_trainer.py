from abc import ABC
from typing import Callable

import torch
from torch import nn, optim, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler

from model.base_trainer import BaseTrainer


def loss(logits: torch.Tensor, labels: torch.Tensor):
    """
    logits: [B, 63]
    labels: [B, 21, 3]
    """

    loss = nn.MSELoss(reduction='sum')(logits, labels)
    # TODO: Add loss for too large joints / regularization -> So the model doesnt go local minimum setting all points 0
    return loss


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
