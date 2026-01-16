import torch
import torch.nn as nn
import torchvision.models as models


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



class CNNLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=256, lstm_layers=1):
        super(CNNLSTM, self).__init__()

        # 1. Define the CNN (Feature Extractor)
        # We use a pretrained ResNet50 for this example
        resnet = models.resnet50(pretrained=True)

        # Remove the final classification layer (fc) so we get the feature vector
        # ResNet50's feature vector size before 'fc' is 2048
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # 2. Define the LSTM
        # Input size is 2048 (from ResNet), Hidden size is usually 128/256/512
        self.lstm = nn.LSTM(input_size=2048,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)

        # 3. Define the Final Classification Layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time_Steps, Channels, Height, Width)
        batch_size, time_steps, C, H, W = x.size()

        # --- CNN STEP ---
        # Reshape to (Batch * Time_Steps, C, H, W) so the CNN treats them as independent images
        c_in = x.view(batch_size * time_steps, C, H, W)

        # Pass through CNN
        c_out = self.cnn(c_in)  # Shape: (Batch * Time, 2048, 1, 1)

        # Flatten the CNN output
        c_out = c_out.view(c_out.size(0), -1)  # Shape: (Batch * Time, 2048)

        # --- LSTM STEP ---
        # Reshape back to (Batch, Time_Steps, Features) for the LSTM
        r_in = c_out.view(batch_size, time_steps, -1)

        # Pass through LSTM
        # r_out shape: (Batch, Time_Steps, Hidden_Size)
        # hidden shape: (Layers, Batch, Hidden_Size) - we usually ignore this here
        r_out, (h_n, c_n) = self.lstm(r_in)

        # --- CLASSIFICATION STEP ---
        # We typically take the output of the LAST time step for classification
        # r_out[:, -1, :] selects the last time step for every batch
        final_out = self.fc(r_out[:, -1, :])

        return final_out


# Example Usage
if __name__ == "__main__":
    # Example: Batch of 4 videos, 10 frames each, 3 channels (RGB), 224x224 res
    dummy_input = torch.rand(4, 10, 3, 224, 224)

    # Initialize model for 5 classes (e.g., walking, running, sitting, etc.)
    model = CNNLSTM(num_classes=5)

    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")  # Should be (4, 5)