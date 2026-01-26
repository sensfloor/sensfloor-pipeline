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

    @abstractmethod
    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

                # 2. Calculate loss
                loss = self.calculate_loss(outputs, labels_on_device)

                # 3. Optimizer zero grad
                self.optimizer.zero_grad()

                # 4. Backward pass and optimization / Loss backward (backpropagation)
                loss.backward()

                # 5. Optimizer step (gradient descent)
                self.optimizer.step()

                # Calculate training accuracy
                total_train_loss += loss.item()
                batch_metrics = self.calculate_metrics(outputs, labels_on_device)
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value

                progress.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)
            train_metrics = {key: value / len(train_loader) for key, value in epoch_metrics.items()}

            avg_val_loss, val_metrics = self.evaluate(validation_loader)

            log_data = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }

            trackio.log(log_data)

            self.epochs_metrics_list.append({"epoch": epoch, **log_data})

            log_data_str = "\n".join([f"{key.upper()}: {value:.4f}" for key, value in log_data.items()])
            print(log_data_str)

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

    def save_metrics_to_csv(self) -> None:
        if not self.epochs_metrics_list:
            return

        csv_path = self.results_path / "metrics.csv"
        keys = self.epochs_metrics_list[0].keys()

        with csv_path.open(mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.epochs_metrics_list)

    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()

        total_loss = 0.0
        total_metrics = defaultdict(float)

        with torch.no_grad():
            for inputs, labels in loader:
                inputs_on_device, labels_on_device = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward_pass(inputs_on_device)
                loss = self.calculate_loss(outputs, labels_on_device)

                total_loss += loss.item()
                batch_metrics = self.calculate_metrics(outputs, labels_on_device)
                for key, value in batch_metrics.items():
                    total_metrics[key] += value

        results = {key: value / len(loader) for key, value in total_metrics.items()}
        return total_loss / len(loader), results
