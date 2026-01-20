import torch
import torch.nn as nn


class RegressionReducedDim(nn.Module):
    """
    idea from Yiyue Luo et. all - Intelligent Carpet: Inferring 3D Human Pose from Tactile Signals
    """

    def __init__(self, history_len: int):
        super().__init__()

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
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )

        self.encoder = nn.Sequential(self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4, self.encoder_5,
                                     self.encoder_6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x


class LSTMStackModel(nn.Module):
    def __init__(self, feature_size: int, landmarks_out: int, lstm_hidden: int = 128,
                 dense_units: int = 64, dense_layers: int = 4):
        """
        feature_size: number of features per time-step (sensor vector length)
        lstm_hidden: hidden size of LSTM (20)
        dense_units: number of neurons in each dense layer (20)
        dense_layers: number of dense layers after LSTM (4)
        """
        super().__init__()
        # LSTM: batch_first=True so input is [batch, seq_len, feature_size]
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)

        # build dense stack
        dense_seq = []
        in_features = lstm_hidden
        for _ in range(dense_layers):
            dense_seq.append(nn.Linear(in_features, dense_units))
            dense_seq.append(nn.ReLU(inplace=True))
            in_features = dense_units

        self.denses = nn.Sequential(*dense_seq)
        self.head = nn.Linear(in_features, landmarks_out)  # single neuron for regression

    def forward(self, x, h_c: tuple | None = None):
        """
        x: [batch, seq_len=window_size, feature_size]
        """
        out, (h_n, c_n) = self.lstm(x, h_c)  # Input None for first state and training # out: [batch, seq_len, lstm_hidden]

        # take last timestep output as representation
        last = out[:, -1, :]  # [batch, lstm_hidden]

        feats = self.denses(last)  # [batch, dense_units]
        logits = self.head(feats)  # [batch, out_dim]
        return logits, (h_n, c_n)


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int, roi_shape: tuple[int, int], lstm_hidden: int = 128,
                 dense_units: int = 64, dense_layers: int = 4, return_hidden_states: bool = False): # TODO try different parameter inputs
        super(CNNLSTM, self).__init__()

        self.return_hidden_states = return_hidden_states

        self.cnn = RegressionReducedDim(history_len=1)

        with torch.no_grad():
            dummy_input = torch.rand((1, 1, *roi_shape))
            cnn_features_out = self._forward_cnn(dummy_input).numel()

        self.lstm = LSTMStackModel(cnn_features_out, num_classes, lstm_hidden=lstm_hidden,
                                   dense_units=dense_units,
                                   dense_layers=dense_layers)

    def _forward_cnn(self, x) -> torch.Tensor:
        # x shape: (Batch, Channels, Height, Width)
        batch_size, history, h, w = x.size()

        # Reshape to (Batch * Time_Steps, C, H, W) so the CNN treats them as independent images
        c_in = x.view(batch_size * history, 1, h, w)

        # Pass through CNN
        c_out = self.cnn(c_in)  # Shape: (Batch * Time, 2048, 1, 1)

        # Flatten the CNN output
        c_out = c_out.view(c_out.size(0), -1)  # Shape: (Batch * Time, 2048)

        # --- LSTM STEP ---
        # Reshape back to (Batch, Time_Steps, Features) for the LSTM
        r_in = c_out.view(batch_size, history, -1)

        return r_in

    def forward(self, x, h_c: tuple | None = None):
        r_in = self._forward_cnn(x)
        r_out, (h_n, c_n) = self.lstm(r_in, h_c)  # Input None for first state and training

        if self.return_hidden_states:
            return r_out, (h_n, c_n)
        else:
            return r_out


# Example Usage
if __name__ == "__main__":
    # Example: Batch of 32, 25 history, 12x12 res
    dummy_input = torch.rand(32, 25, 12, 12)

    # model = RegressionModel(landmarks_out=99, history_len=25, roi_shape=(12,12))
    model = CNNLSTM(num_classes=13, roi_shape=(12, 12), dense_units=20, dense_layers=4, lstm_hidden=20)

    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
