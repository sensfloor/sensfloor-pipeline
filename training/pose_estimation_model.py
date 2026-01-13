import torch
from torch import nn
from training.utils import SIGNAL_Z


class RegressionModel(nn.Module):
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


class RegressionModelMaxPool(nn.Module):
    """
    Model from Yiyue Luo et. all - Intelligent Carpet: Inferring 3D Human Pose from Tactile Signals
    """

    def __init__(self, roi_shape: tuple[int, int], history_len: int, landmarks_out: int):
        super().__init__()

        # 12 x 12 # history_len
        self.conv_0 = nn.Sequential(
            nn.Conv2d(2 * history_len, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))

        # 6 x 6 x history_len

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            #    nn.MaxPool2d(kernel_size=2) # 4 x 4 x history_len
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))

        self.conv_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            # nn.MaxPool2d(kernel_size=2)
        )

        self.encoder = nn.Sequential(self.conv_0, self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5,
                                     self.conv_6)

        dummy_input = torch.zeros((1, history_len, roi_shape[0], roi_shape[1]))
        linear_in_features = self.encoder(dummy_input).numel()
        self.linear = nn.Linear(in_features=linear_in_features, out_features=landmarks_out * 3)  # 3 coordinates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class HeatMapSigmoidModel(nn.Module):
    """
    idea from Yiyue Luo et. all - Intelligent Carpet: Inferring 3D Human Pose from Tactile Signals
    """

    def __init__(self):
        # TODO: add some kind of max pooling? Didn't because our input is small, but maybe at least padding remove once or sth
        # TODO: adapt kernel size, because we have a smaller input?
        super().__init__()

        # 4x4x64
        encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        )
        encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )
        encoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )
        encoder_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )
        encoder_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )
        encoder_6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
        )
        encoder_7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
        )

        self.encoder = nn.Sequential(encoder_1, encoder_2, encoder_3, encoder_4, encoder_5, encoder_6, encoder_7)

        decoder_1 = nn.Sequential(
            nn.Conv3d(in_channels=1025, out_channels=1025, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(1025),
        )
        decoder_2 = nn.Sequential(
            nn.Conv3d(in_channels=1025, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(512),
        )
        decoder_3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm3d(256),
        )
        decoder_4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128),
        )
        decoder_5 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64),
        )
        decoder_6 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=21, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )

        self.decoder = nn.Sequential(decoder_1, decoder_2, decoder_3, decoder_4, decoder_5, decoder_6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)

        # --- Transform into 3D ---
        B, C, X, Y = x.shape
        Z = int(SIGNAL_Z / 2)

        x = x.unsqueeze(-1).repeat(1, 1, 1, 1, Z)  # [B, C, X, Y, Z]

        z_index = torch.arange(Z, dtype=x.dtype, device=x.device)
        z_index = z_index.view(1, 1, 1, 1, Z)
        z_index = z_index.repeat(B, 1, X, Y, 1)  # [B, 1, X, Y, Z]

        # Add 1 channel for height (so that the model can differentiate between the repeated layers
        x = torch.cat([x, z_index], dim=1)  # [B, C+1, X, Y, Z]

        x = self.decoder(x)

        return x
