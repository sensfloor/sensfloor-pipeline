import csv
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from src.definitions import ROOT_PATH

METRICS_FILE_NAME = "metrics.csv"

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


    def save_metrics_to_csv(self) -> None:
        if not self.epochs_metrics_list:
            return

        csv_path = self.results_path / METRICS_FILE_NAME
        keys = self.epochs_metrics_list[-1].keys()

        with csv_path.open(mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.epochs_metrics_list)