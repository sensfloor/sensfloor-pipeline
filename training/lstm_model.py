import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- sliding window utility ----------
def sliding_windows(sequence: torch.Tensor, window_size: int = 30, step: int = 1):
    """
    Convert a 2D tensor [T, features] into windows of shape [num_windows, window_size, features].
    """
    T, F = sequence.shape
    if T < window_size:
        return sequence.unsqueeze(0)  # single, short window (you may want to pad instead)
    indices = range(0, T - window_size + 1, step)
    windows = [sequence[i:i + window_size] for i in indices]
    return torch.stack(windows, dim=0)  # [num_windows, window_size, F]


# ---------- example dataset using sliding windows ----------
class SlidingWindowDataset(Dataset):
    def __init__(self, raw_signals: list[torch.Tensor], labels: list[torch.Tensor],
                 window_size: int = 30, step: int = 1):
        """
        raw_signals: list of T x F tensors (each recording)
        labels: list of per-window labels or per-recording labels; this constructor assumes per-window labels
                If your labels are per-recording, you'll need to map them to windows.
        """
        self.windows = []
        self.window_labels = []

        for signal, lab in zip(raw_signals, labels):
            # signal: [T, F], lab: either [T'] or single label - adapt as required
            w = sliding_windows(signal, window_size=window_size, step=step)  # [Nw, window_size, F]
            # For simplicity, assume lab is a per-window label tensor of length Nw
            # If lab is a single label for the whole recording, expand it: lab_expand = lab.repeat(len(w), ...)
            if lab.ndim == 1 and lab.shape[0] == w.shape[0]:
                self.window_labels.append(lab)
            else:
                # fallback: assume one label per recording
                lab_expand = lab.unsqueeze(0).expand(w.shape[0], *lab.shape[1:])
                self.window_labels.append(lab_expand)

            self.windows.append(w)

        self.windows = torch.cat(self.windows, dim=0)
        self.window_labels = torch.cat(self.window_labels, dim=0)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.window_labels[idx]

class LSTMStackModel(nn.Module):
    def __init__(self, feature_size: int, landmarks_out: int, window_size: int = 25, lstm_hidden: int = 20,
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

        self.head = nn.Linear(in_features, landmarks_out)   # single neuron for regression
        self.final_act = nn.ReLU()              # TODO apply ReLU if paper specifies it

    def forward(self, x):
        """
        x: [batch, seq_len=window_size, feature_size]
        returns:
            classification: logits shape [batch, 2]
            regression: non-negative scalar per sample [batch, 1] (due to ReLU)
        """
        # LSTM
        out, (h_n, c_n) = self.lstm(x)               # out: [batch, seq_len, lstm_hidden]
        # take last timestep output as representation
        last = out[:, -1, :]                         # [batch, lstm_hidden]

        # dense stack
        feats = self.denses(last)                    # [batch, dense_units]

        logits = self.head(feats)                    # [batch, out_dim]
        return self.final_act(logits)                # apply ReLU for regression


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device).float()          # [B, seq_len, F]
        targets = targets.to(device).long() if model.task=='classification' else targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)                      # [B, 2] logits or [B,1]
        if model.task == 'classification':
            # targets should be integer class labels (0/1) of shape [B] or [B] with dtype long
            # if your labels are one-hot, convert: targets = targets.argmax(dim=1)
            loss = loss_fn(outputs, targets)
        else:
            # regression: outputs [B,1], targets shape must match e.g. [B,1]
            loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).long() if model.task=='classification' else targets.to(device).float()
            outputs = model(inputs)

            if model.task == 'classification':
                loss = loss_fn(outputs, targets)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
            else:
                loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = None
    if model.task == 'classification':
        acc = correct / len(dataloader.dataset)
    return avg_loss, acc


# ---------- Example usage ----------
if __name__ == "__main__":
    # toy example: 64-dimensional sensor vector per timestep
    feature_size = 64
    window_size = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create synthetic data: list of recordings, each T x F
    # Here we create two recordings of length 200 with random data
    recordings = [torch.randn(200, feature_size), torch.randn(180, feature_size)]
    # per-window labels: classification with 0/1 per window (random here)
    labels_per_record = []
    for rec in recordings:
        Nw = rec.shape[0] - window_size + 1
        labels_per_record.append(torch.randint(0, 2, (Nw,)))  # binary labels per window

    dataset = SlidingWindowDataset(recordings, labels_per_record, window_size=window_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMStackModel(feature_size=feature_size, window_size=window_size,
                           lstm_hidden=20, dense_units=20, dense_layers=4,
                           task='classification').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # one epoch
    train_loss = train_epoch(model, loader, optimizer, loss_fn, device)
    val_loss, val_acc = eval_epoch(model, loader, loss_fn, device)
    print("train_loss", train_loss, "val_loss", val_loss, "val_acc", val_acc)