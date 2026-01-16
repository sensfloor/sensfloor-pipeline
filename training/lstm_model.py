import torch
from torch import nn


class LSTMStackModel(nn.Module):
    def __init__(self, feature_size: int, landmarks_out: int, window_size: int, lstm_hidden: int = 20,
                 dense_units: int = 20, dense_layers: int = 4, ):
        """
        feature_size: number of features per time-step (sensor vector length)
        window_size: length of input sequence (30)
        lstm_hidden: hidden size of LSTM (20)
        dense_units: number of neurons in each dense layer (20)
        dense_layers: number of dense layers after LSTM (4)
        task: 'classification' or 'regression'
        """
        super().__init__()
        self.window_size = window_size

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
        self.final_act = nn.ReLU()  # TODO apply ReLU if paper specifies it

    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len=window_size, feature_size]
        returns:
            classification: logits shape [batch, 2]
            regression: non-negative scalar per sample [batch, 1] (due to ReLU)
        """
        b, hist, roi_x, roi_y = x.shape

        x = x.view((b, hist, roi_x * roi_y))

        # LSTM
        out, (h_n, c_n) = self.lstm(x)  # out: [batch, seq_len, lstm_hidden]
        # take last timestep output as representation
        last = out[:, -1, :]  # [batch, lstm_hidden]

        # dense stack
        feats = self.denses(last)  # [batch, dense_units]

        logits = self.head(feats)  # [batch, out_dim]
        return self.final_act(logits)  # apply ReLU for regression
