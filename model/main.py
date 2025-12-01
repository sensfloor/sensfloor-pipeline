import torch
from torch import nn
from torch.utils.data import DataLoader

from SensfloorTrainer import SensfloorTrainer
from pose_estimation_model import RegressionModel
from pose_dataset import PoseDataset
from utils import set_seed

set_seed(seed=42)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

model = RegressionModel(input_shape=(12, 12))

dataset_train = PoseDataset()
dataset_val = PoseDataset()
train_loader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=32)
val_loader = DataLoader(dataset=dataset_val, shuffle=True, batch_size=32)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

trainer = SensfloorTrainer(model=model, device=device, optimizer=optimizer, patience=5, use_early_stopping=True,)
trainer.train(train_loader=train_loader, validation_loader=val_loader, epochs=10)
