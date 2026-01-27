import csv
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import trackio
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from definitions import ROOT_PATH


class BaseTrainer(metaclass=ABCMeta):
    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: LRScheduler | None = None,
        use_early_stopping: bool = True,
        patience: int = 10,
        results_path: Path = ROOT_PATH,
        best_model_name: str = "best_model.pth",
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.optimizer: optim.Optimizer = optimizer
        self.device: torch.device = device
        self.scheduler: LRScheduler | None = scheduler
        self.use_early_stopping: bool = use_early_stopping
        self.patience: int = patience
        self.best_model_name: str = best_model_name
        self.best_val_loss: float = float("inf")
        self.patience_counter: float = 0
        self.epochs_metrics_list: list[dict[str, float]] = []
        self.results_path: Path = results_path
        self.results_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pass

    @abstractmethod
    def calculate_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        pass

    def early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            print("Early stopping triggered!")
            return True
        return False

    def save_best_model(self, val_loss: float) -> None:
        if val_loss < self.best_val_loss:
            torch.save(self.model.state_dict(), self.results_path / self.best_model_name)
            print(f"Best model updated with validation loss {val_loss:.4f}.")

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
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }

            # Logging
            trackio.log(log_data)
            self.epochs_metrics_list.append(log_data)
            self.log_formatted_table(log_data)

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

    def log_formatted_table(self, log_data: dict[str, float]) -> None:
        """
        Prints metrics in a table format:
        Row 1: Metric Name (e.g. Train MJPE)
        Col 1: Mean
        Col 2-N: Individual Joints
        """
        # 1. Header & Loss Section
        separator = "-" * 200
        print("\n" + "=" * 120)
        print(f"EPOCH {log_data.get('epoch', '?')} SUMMARY")
        print(f"Losses | Train: {log_data.get('train_loss', 0):.4f} | "
              f"Val: {log_data.get('val_loss', 0):.4f}, "
              f"MSE: {log_data.get('val_loss_mse', 0):.4f}, "
              f"Link: {log_data.get('val_loss_link', 0):.4f}")
        print(separator)

        # 2. Identify Joints dynamically
        # We look for keys starting with 'val_mjpe_' to find the joint names
        joint_names = [
            k.replace("val_mjpe_", "")
            for k in log_data.keys()
            if k.startswith("val_mjpe_") and "_mean" not in k
        ]

        if not joint_names:
            return

        # 3. Create Short Headers for Columns (e.g., LEFT_SHOULDER -> L_SHOU)
        # 8 chars width per column usually fits standard terminals
        headers = ["METRIC", "MEAN"] + [
            name.replace("LEFT", "L").replace("RIGHT", "R")[:6] for name in joint_names
        ]

        # Define spacing format: First col 12 wide, others 8 wide
        row_fmt = "{:<12} " + "{:>8} " * (len(headers) - 1)

        print(row_fmt.format(*headers))
        print(separator)

        # 4. Helper to print a specific metric row
        def print_metric_row(display_name, prefix):
            # Get Mean
            mean_val = log_data.get(f"{prefix}_mean", 0.0)
            # Get Joint Values
            joint_vals = [log_data.get(f"{prefix}_{name}", 0.0) for name in joint_names]

            # Combine
            all_vals = [mean_val] + joint_vals

            # Format numbers (remove 0. if it's 0.0000 to save space, optional)
            formatted_vals = [f"{v:.4f}" for v in all_vals]
            print(row_fmt.format(display_name, *formatted_vals))

        # 5. Print the rows
        print_metric_row("Train MJPE", "train_mjpe")
        print_metric_row("Val MJPE", "val_mjpe")
        print(separator)
        print_metric_row("Train PCK10", "train_pck_10")
        print_metric_row("Val PCK10", "val_pck_10")
        print(separator)
        print_metric_row("Train PCK5", "train_pck_5")
        print_metric_row("Val PCK5", "val_pck_5")
        print(separator + "\n")

    def save_metrics_to_csv(self) -> None:
        if not self.epochs_metrics_list:
            return

        csv_path = self.results_path / "metrics.csv"
        keys = self.epochs_metrics_list[-1].keys()

        with csv_path.open(mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.epochs_metrics_list)

    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_metrics = defaultdict(float)

        with torch.no_grad():
            for inputs, labels in loader:
                inputs_on_device, labels_on_device = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward_pass(inputs_on_device)

                loss, loss_dict = self.calculate_loss(outputs, labels_on_device)

                total_metrics["total_loss"] += loss.item()
                for key, value in loss_dict.items():
                    total_metrics[key] += value

                batch_metrics = self.calculate_metrics(outputs, labels_on_device)
                for key, value in batch_metrics.items():
                    total_metrics[key] += value

        results = {key: value / len(loader) for key, value in total_metrics.items()}
        return results["total_loss"], results