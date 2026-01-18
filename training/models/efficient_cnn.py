import torch
import torch.nn as nn


class RegressionReducedDim(nn.Module):
    """
    idea from Yiyue Luo et. all - Intelligent Carpet: Inferring 3D Human Pose from Tactile Signals
    """

    def __init__(self, roi_shape: tuple[int, int], history_len: int, landmarks_out: int):
        # TODO: adapt kernel size, because we have a smaller input?
        # TODO: consider removing batchnorm, because we want to have value predictions (no sigmoid or classification)
        # TODO: consider switching ReLU and Batchnorm
        super().__init__()

        # 4x4x64
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=history_len, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )

        self.encoder = nn.Sequential(self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4, self.encoder_5,
                                     self.encoder_6)

        dummy_input = torch.zeros((1, history_len, roi_shape[0], roi_shape[1]))
        linear_in_features = self.encoder(dummy_input).numel()
        self.linear = nn.Linear(in_features=linear_in_features, out_features=landmarks_out * 3)  # 3 coordinates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

class RegressionModelMaxPool(nn.Module):
    """
    idea from Yiyue Luo et. all - Intelligent Carpet: Inferring 3D Human Pose from Tactile Signals
    """

    def __init__(self, roi_shape: tuple[int, int], history_len: int, landmarks_out: int):
        # TODO: adapt kernel size, because we have a smaller input?
        # TODO: consider removing batchnorm, because we want to have value predictions (no sigmoid or classification)
        # TODO: consider switching ReLU and Batchnorm
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
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
        )
        self.encoder_7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
        )

        self.encoder = nn.Sequential(self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4, self.encoder_5,
                                     self.encoder_6, self.encoder_7)

        dummy_input = torch.zeros((1, history_len, roi_shape[0], roi_shape[1]))
        linear_in_features = self.encoder(dummy_input).numel()
        self.linear = nn.Linear(in_features=linear_in_features, out_features=landmarks_out * 3)  # 3 coordinates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

class RegressionModelBatchnormFirst(nn.Module):
    """
    idea from Yiyue Luo et. all - Intelligent Carpet: Inferring 3D Human Pose from Tactile Signals
    """

    def __init__(self, roi_shape: tuple[int, int], history_len: int, landmarks_out: int):
        # TODO: adapt kernel size, because we have a smaller input?
        # TODO: consider removing batchnorm, because we want to have value predictions (no sigmoid or classification)
        super().__init__()

        # 4x4x64
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=history_len, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding="valid"),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.encoder_7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )

        self.encoder = nn.Sequential(self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4, self.encoder_5,
                                     self.encoder_6, self.encoder_7)

        dummy_input = torch.zeros((1, history_len, roi_shape[0], roi_shape[1]))
        linear_in_features = self.encoder(dummy_input).numel()
        self.linear = nn.Linear(in_features=linear_in_features, out_features=landmarks_out * 3)  # 3 coordinates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

class RegressionModelNoBatchnorm(nn.Module):
    """
    idea from Yiyue Luo et. all - Intelligent Carpet: Inferring 3D Human Pose from Tactile Signals
    """

    def __init__(self, roi_shape: tuple[int, int], history_len: int, landmarks_out: int):
        # TODO: adapt kernel size, because we have a smaller input?
        super().__init__()

        # 4x4x64
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=history_len, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="valid"),
            nn.LeakyReLU(),
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="valid"),
            nn.LeakyReLU(),
        )
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding="valid"),
            nn.LeakyReLU(),
        )
        self.encoder_7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding="valid"),
            nn.LeakyReLU(),
        )

        self.encoder = nn.Sequential(self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4, self.encoder_5,
                                     self.encoder_6, self.encoder_7)

        dummy_input = torch.zeros((1, history_len, roi_shape[0], roi_shape[1]))
        linear_in_features = self.encoder(dummy_input).numel()
        self.linear = nn.Linear(in_features=linear_in_features, out_features=landmarks_out * 3)  # 3 coordinates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x