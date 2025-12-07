from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


@dataclass
class TrainingMetrics:
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch: int


class BaseTrainer(metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss: _Loss | Callable[[Tensor, Tensor], Tensor],
        device: torch.device,
        scheduler: LRScheduler | None = None,
        use_early_stopping: bool = True,
        patience: int = 10,
        best_model_name: str = "best_model.pth",
    ):
        self.model: nn.Module = model.to(device)
        self.optimizer: optim.Optimizer = optimizer
        self.loss: _Loss | Callable[[Tensor, Tensor], Tensor] = loss
        self.device: torch.device = device
        self.scheduler: LRScheduler | None = scheduler
        self.use_early_stopping: bool = use_early_stopping
        self.patience: int = patience
        self.best_model_name: str = best_model_name
        self.best_val_loss: float = float("inf")
        self.patience_counter: float = 0
        self.metrics_list: list[Any] = []

    @abstractmethod
    def forward_pass(self, inputs: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def calculate_loss(self, outputs: Any, labels: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_accuracy(self, outputs: Any, labels: Any) -> float:
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
            torch.save(self.model.state_dict(), self.best_model_name)
            print(f"Best model updated with validation loss {val_loss:.4f}.")

    def train(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int) -> None:
        for epoch in range(epochs):
            self.model.train()
            total_loss: float = 0.0
            total_accuracy: float = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 1. Forward pass
                outputs: Any = self.forward_pass(inputs)

                # 2. Calculate loss
                loss = self.calculate_loss(outputs, labels)

                # 3. Optimizer zero grad
                self.optimizer.zero_grad()

                # 4. Backward pass and optimization / Loss backward (backpropagation)
                loss.backward()

                # 5. Optimizer step (gradient descent)
                self.optimizer.step()

                # Calculate training accuracy
                total_loss += loss.item()
                total_accuracy += self.calculate_accuracy(outputs, labels)

            avg_loss: float = total_loss / len(train_loader)
            avg_accuracy: float = total_accuracy / len(train_loader)
            val_loss, val_accuracy = self.evaluate(validation_loader)

            self.metrics_list.append(
                TrainingMetrics(
                    train_loss=avg_loss,
                    train_accuracy=avg_accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    epoch=epoch + 1,
                ),
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_accuracy:.4f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}%",
            )

            self.save_best_model(val_loss)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                    self.scheduler.step()

            if self.use_early_stopping and self.early_stopping(val_loss):
                break

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss: float = 0.0
        total_accuracy: float = 0.0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward_pass(inputs)
                loss = self.calculate_loss(outputs, labels)
                total_loss += loss.item()
                total_accuracy += self.calculate_accuracy(outputs, labels)

        return total_loss / len(loader), total_accuracy / len(loader)
