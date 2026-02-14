import torch
from torch import nn


class RegressionModel(nn.Module):
    """
    idea from Yiyue Luo et. all - Intelligent Carpet: Inferring 3D Human Pose from Tactile Signals
    """

    def __init__(self, roi_shape: tuple[int, int], history_len: int, landmarks_out: int):
        super().__init__()

        # 4x4x64
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=history_len, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="valid"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="valid"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding="valid"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
        )
        self.encoder_7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding="valid"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
        )

        self.encoder = nn.Sequential(
            self.encoder_1,
            self.encoder_2,
            self.encoder_3,
            self.encoder_4,
            self.encoder_5,
            self.encoder_6,
            self.encoder_7,
        )

        dummy_input = torch.zeros((1, history_len, roi_shape[0], roi_shape[1]))
        linear_in_features = self.encoder(dummy_input).numel()
        self.linear = nn.Linear(in_features=linear_in_features, out_features=landmarks_out * 3)  # 3 coordinates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
