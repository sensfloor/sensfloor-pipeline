from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from data_loading.pose_landmark import PoseLandmark
from definitions import ROOT_PATH
from training.link_loss.link_loss import calculate_linkloss
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
    def calculate_mean_accuracy(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        landmarks_out: int,
        threshold: float = 0.1,
    ) -> float:
        joint_accuracy = SensfloorTrainer.calculate_percentage_correct_keypoints(
            outputs,
            labels,
            landmarks_out,
            threshold,
        )
        accuracy = joint_accuracy.mean()  # average over all B × landmarks_out
        return accuracy.item() * 100

    @staticmethod
    def get_distances(outputs: torch.Tensor,labels: torch.Tensor,landmarks_out: int):
        joint_coordinates = outputs.view(-1, landmarks_out, 3)  # [B, landmarks_out, 3]
        ground_truth = labels.view(-1, landmarks_out, 3)  # [B, landmarks_out, 3]
        return torch.linalg.vector_norm(joint_coordinates - ground_truth, dim=2)  # [B, landmarks_out]

    @staticmethod
    def calculate_percentage_correct_keypoints(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        landmarks_out: int,
        threshold: float,
    ) -> torch.Tensor:
        distances = SensfloorTrainer.get_distances(outputs, labels, landmarks_out)
        correct = distances < threshold  # values: [True, False, ...]
        return correct.float()  # values: [1, 0, ...]

    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mse_loss = nn.MSELoss(reduction="mean")(outputs, labels)
        link_loss = (
            calculate_linkloss(outputs, self.k_min, self.k_max, self.pose_to_model_dict, links=self.kept_links)
            * self.amplify_link_loss
        )  # Paper amplifies link loss by 10
        return mse_loss + link_loss

    def calculate_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        distances = SensfloorTrainer.get_distances(outputs, labels, self.landmarks_out)
        # Mean joint position error
        mjpe = distances.mean().item()

        # Percentage correct keypoints
        pck_10 = (
            self.calculate_percentage_correct_keypoints(
                outputs=outputs,
                labels=labels,
                landmarks_out=self.landmarks_out,
                threshold=0.1,
            )
            .mean()
            .item()
        )

        pck_5 = (
            self.calculate_percentage_correct_keypoints(
                outputs=outputs,
                labels=labels,
                landmarks_out=self.landmarks_out,
                threshold=0.05,
            )
            .mean()
            .item()
        )
        return {
            "mjpe": mjpe,
            "pck_10": pck_10,
            "pck_5": pck_5,
        }


def get_test_accuracy(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    trainer_instance: SensfloorTrainer,
) -> dict[str, float]:
    model.eval()
    model.to(device)
    total_metrics: dict[str, float] = defaultdict(float)
    num_batches = len(test_loader)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs_on_device, labels_on_device = inputs.to(device), labels.to(device)
            outputs = model(inputs_on_device)

            batch_metrics = trainer_instance.calculate_metrics(outputs, labels_on_device)

            for key, value in batch_metrics.items():
                total_metrics[key] += value

    return {f"test_{key}": value / num_batches for key, value in total_metrics.items()}
