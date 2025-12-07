from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from model.base_trainer import BaseTrainer
from model.link_loss import calculate_linkloss
from data_loading.links_min_max import compute_kmin_kmax

class SensfloorTrainer(BaseTrainer):
    def __init__(self, model: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 links_path: Path,
                 scheduler: LRScheduler | None = None,
                 use_early_stopping: bool = True,
                 patience: int = 10,
                 best_model_name: str = "best_model.pth"):
        super().__init__(model=model, optimizer=optimizer, device=device, scheduler=scheduler,
                         use_early_stopping=use_early_stopping, patience=patience, best_model_name=best_model_name)

        self.k_min, self.k_max = compute_kmin_kmax(csv_path=links_path)


    def forward_pass(self, inputs: torch.Tensor):
        return self.model(inputs)

    def calculate_loss(self, outputs, labels) -> torch.Tensor:
        mse_loss = nn.MSELoss(reduction='mean')(outputs, labels)
        link_loss = calculate_linkloss(outputs, self.k_min, self.k_max) / len(self.k_min)
        loss = mse_loss + link_loss
        # TODO: Add loss for too large joints / regularization -> So the model doesnt go local minimum setting all points 0
        return loss

    def calculate_accuracy(self, outputs, labels, threshold=0.1):
        coords = outputs.view(-1, 17, 3)  # [B, 17, 3] #TODO: parametrize landmarks_out
        reshaped_labels = labels.view(-1, 17, 3)  # [B, 17, 3] #TODO: parametrize landmarks_out
        dist = torch.linalg.vector_norm(coords - reshaped_labels, dim=2)  # [B, 17]
        correct = (dist < threshold)
        accuracy = correct.float().mean()  # average over all B × 17
        return accuracy * 100
