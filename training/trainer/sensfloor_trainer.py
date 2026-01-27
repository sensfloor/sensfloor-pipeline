from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trackio
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loading.pose_landmark import PoseLandmark
from definitions import ROOT_PATH
from training.link_loss.link_loss import calculate_linkloss
from training.trainer.base_trainer import BaseTrainer

# Metrics Strings also used for Logging
VAL_PREFIX = "val_"
TRAIN_PREFIX = "train_"
VAL_LOSS = "val_loss"
TRAIN_LOSS = "train_loss"
EPOCH = "epoch"
LOSS_LINK = "loss_link"
LOSS_MSE = "loss_mse"
MJPE_PREFIX = "mjpe_"
MEAN = "mean"
PCK_THRESHOLDS = {"pck_10_": 0.1, "pck_5_": 0.05}


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
        # Create a reverse mapping (int -> Name) for easy logging
        self.idx_to_name = {v: k.name for k, v in self.pose_to_model_dict.items()}

    @staticmethod
    def get_distances(outputs: torch.Tensor, labels: torch.Tensor, landmarks_out: int):
        joint_coordinates = outputs.view(-1, landmarks_out, 3)  # [B, landmarks_out, 3]
        ground_truth = labels.view(-1, landmarks_out, 3)  # [B, landmarks_out, 3]
        return torch.linalg.vector_norm(joint_coordinates - ground_truth, dim=2)  # [B, landmarks_out]

    @staticmethod
    def calculate_percentage_correct_keypoints(
            distances: torch.Tensor,
            threshold: float,
    ) -> torch.Tensor:
        """Returns tensor of shape [B, landmarks_out] with 1.0 for correct, 0.0 for incorrect."""
        correct = distances < threshold
        return correct.float()

    def train(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int) -> None:
        for epoch in range(epochs):
            num_batches = len(train_loader)
            progress = tqdm(
                iterable=enumerate(train_loader),
                total=num_batches,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
            )
            self.model.train()

            total_train_loss: float = 0.0
            epoch_metrics: dict[str, float] = defaultdict(float)

            for _, (inputs, labels) in progress:
                inputs_on_device, labels_on_device = inputs.to(self.device), labels.to(self.device)

                # 1. Forward pass
                outputs: Any = self.forward_pass(inputs_on_device)

                # 2. Calculate loss (Now unpacks tuple)
                loss, _ = self.calculate_loss(outputs, labels_on_device)

                # 3. Optimizer zero grad
                self.optimizer.zero_grad()

                # 4. Backward pass
                loss.backward()

                # 5. Optimizer step
                self.optimizer.step()

                total_train_loss += loss.item()

                # Aggregate Performance Metrics
                batch_metrics = self.calculate_metrics(outputs, labels_on_device)
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value

                progress.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Average over batches
            avg_train_loss = total_train_loss / len(train_loader)
            train_metrics = {key: value / len(train_loader) for key, value in epoch_metrics.items()}

            avg_val_loss, val_metrics = self.evaluate(validation_loader)

            log_data = {
                EPOCH: epoch + 1,
                TRAIN_LOSS: avg_train_loss,
                VAL_LOSS: avg_val_loss,
                **{f"{TRAIN_PREFIX}{key}": value for key, value in train_metrics.items()},
                **{f"{VAL_PREFIX}{key}": value for key, value in val_metrics.items()},
            }

            # Logging
            trackio.log(log_data)
            self.epochs_metrics_list.append(log_data)
            self.log_training_metrics(log_data)

            self.save_best_model(avg_val_loss)
            self.save_metrics_to_csv()

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                elif isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                    self.scheduler.step()

            if self.use_early_stopping and self.early_stopping(avg_val_loss):
                break

            self.best_val_loss = min(self.best_val_loss, avg_val_loss)

    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        mse_loss = nn.MSELoss(reduction="mean")(outputs, labels)
        link_loss = (
                calculate_linkloss(outputs, self.k_min, self.k_max, self.pose_to_model_dict, links=self.kept_links)
                * self.amplify_link_loss
        )  # Paper amplifies link loss by 10

        total_loss = mse_loss + link_loss

        loss_components = {
            LOSS_MSE: mse_loss.item(),
            LOSS_LINK: link_loss.item(),
        }

        return total_loss, loss_components

    def calculate_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        distances = SensfloorTrainer.get_distances(outputs, labels, self.landmarks_out)
        metrics = {(MJPE_PREFIX + MEAN): distances.mean().item()}
        # Mean joint position error
        per_joint_mjpe = distances.mean(dim=0)  # [Num_Joints]

        for idx, error in enumerate(per_joint_mjpe):
            joint_name = self.idx_to_name.get(idx, f"joint_{idx}")
            metrics[f"{MJPE_PREFIX}{joint_name}"] = error.item()

        for name, thresh in PCK_THRESHOLDS.items():
            correct_matrix = self.calculate_percentage_correct_keypoints(distances, thresh)
            metrics[f"{name}{MEAN}"] = correct_matrix.mean().item()

            # Per Joint Accuracy
            per_joint_acc = correct_matrix.mean(dim=0)  # [Num_Joints]

            for idx, acc in enumerate(per_joint_acc):
                joint_name = self.idx_to_name.get(idx, f"joint_{idx}")
                metrics[f"{name}_{joint_name}"] = acc.item()

        return metrics

    def log_training_metrics(self, log_data: dict[str, float]) -> None:
        """
        Prints metrics in a table format:
        Row 1: Metric Name (e.g. Train MJPE)
        Col 1: Mean
        Col 2-N: Individual Joints
        """
        # Header & Loss Section
        outputs = []
        separator = "-" * 200
        outputs.append("\n" + "=" * 120)
        outputs.append(f"EPOCH {log_data.get(EPOCH, '?')} SUMMARY")
        outputs.append(f"Losses | Train: {log_data.get(TRAIN_LOSS, -1):.4f} | "
                       f"Val: {log_data.get(VAL_LOSS, -1):.4f}, "
                       f"MSE: {log_data.get(VAL_PREFIX + LOSS_MSE, -1):.4f}, "
                       f"Link: {log_data.get(VAL_PREFIX + LOSS_LINK, -1):.4f}")
        outputs.append(separator)

        # Metrics
        joint_names = [k.name for k in self.pose_to_model_dict.keys()]
        headers = ["METRIC", "MEAN"] + [
            name.replace("LEFT", "L").replace("RIGHT", "R")[:6] for name in joint_names
        ]

        # Spacing format: First col 13 wide, others 8 wide
        row_fmt = "{:<13} " + "{:>8} " * (len(headers) - 1)

        outputs.append(row_fmt.format(*headers))
        outputs.append(separator)

        def print_metric_row(display_name, prefix):
            mean_val = log_data.get(f"{prefix}{MEAN}", 0.0)
            joint_vals = [log_data.get(f"{prefix}{name}", 0.0) for name in joint_names]
            all_vals = [mean_val] + joint_vals

            formatted_vals = [f"{v:.4f}" for v in all_vals]
            return row_fmt.format(display_name, *formatted_vals)

        outputs.append(print_metric_row("Train MJPE", f"{TRAIN_PREFIX}{MJPE_PREFIX}"))
        outputs.append(print_metric_row("Val MJPE", f"{VAL_PREFIX}{MJPE_PREFIX}"))

        outputs.append(separator)

        for prefix, value in PCK_THRESHOLDS.items():
            outputs.append(print_metric_row(f"Train {prefix}", f"{TRAIN_PREFIX}{prefix}"))
            outputs.append(print_metric_row(f"Val {prefix}", f"{VAL_PREFIX}{prefix}"))
            outputs.append(separator)

        print("\n".join(outputs))


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
