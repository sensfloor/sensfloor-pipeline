import torch
import torch.nn as nn


class EfficientCNNLSTM(nn.Module):  # TODO Change dataset to have real sequences
    def __init__(self, num_classes: int, roi_shape: tuple[int, int], hidden_size=256, num_layers=2):
        super().__init__()

        self.cnn = nn.Sequential(
            # Layer 1: 12x12 -> 12x12
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2: 12x12 -> 6x6 (Pooling reduces spatial dim, keeps features robust)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3: 6x6 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Flatten size: 128 channels * 3 * 3 = 1152 features

        with torch.no_grad():
            dummy_input = torch.rand((1,1, *roi_shape))
            cnn_out_size = self.cnn(dummy_input).numel()

        # smooth transition from CNN to LSTM
        self.projection = nn.Linear(cnn_out_size, hidden_size)

        # big hidden size for capturing motion
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (Batch, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, T, 1, H, W)

        b, t, c, h, w = x.size()

        # CNN Pass
        c_in = x.view(b * t, c, h, w)
        c_out = self.cnn(c_in)
        c_out = c_out.view(c_out.size(0), -1)  # Flatten -> (B*T, 1152)

        # Projection
        features = self.projection(c_out)  # (B*T, 256)

        # LSTM Pass
        r_in = features.view(b, t, -1)
        r_out, _ = self.lstm(r_in)  # (B, T, 256)

        # Regression on the LAST frame
        last_frame_feat = r_out[:, -1, :]
        pred = self.regressor(last_frame_feat)

        return pred

# Example Usage
if __name__ == "__main__":
    # Example: Batch of 32, 25 history, 12x12 res
    dummy_input = torch.rand(32, 30, 24, 24)

    #model = RegressionModel(landmarks_out=99, history_len=25, roi_shape=(12,12))
    model = EfficientCNNLSTM(num_classes=99, roi_shape=(24,24))

    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")