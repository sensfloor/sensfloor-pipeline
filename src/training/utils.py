import random

import numpy as np
import torch

from src.data_loading.pose_landmark import LINKS
from src.training.configs import ModelType
from src.training.models.CNN.cnn_default import RegressionModel
from src.training.models.CNN.cnn_variants import (
    RegressionModelBatchnormFirst,
    RegressionModelMaxPool,
    RegressionModelNoBatchnorm,
    RegressionReducedDim,
)
from src.training.models.LSTM.efficient_lstm import EfficientCNNLSTM
from src.training.models.LSTM.lstm_model import CNNLSTM


def get_kept_links(drop_landmarks):
    kept_links = []
    for link in LINKS:
        if link[0] in drop_landmarks or link[1] in drop_landmarks:
            continue
        kept_links.append(link)
    return kept_links


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "mps") and torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"using device: {device}")
    return torch.device(device)


def set_seed(seed: int = 42):
    if seed is None:
        seed = random.randint(1, 1000)

    print(f"Random Seed: {seed}")

    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(
    roi_shape: tuple[int, int],
    landmarks_out: int,
    history_maxlen: int,
    model_type: ModelType,
    return_hidden_states: bool = False,
) -> torch.nn.Module:
    match model_type:
        case ModelType.CNN:
            return RegressionModel(
                roi_shape=roi_shape,
                landmarks_out=landmarks_out,
                history_len=history_maxlen,
            )
        case ModelType.CNN_EFFICIENT:
            return RegressionReducedDim(
                roi_shape=roi_shape,
                landmarks_out=landmarks_out,
                history_len=history_maxlen,
            )
        case ModelType.CNN_NO_BATCHNORM:
            return RegressionModelNoBatchnorm(
                roi_shape=roi_shape,
                landmarks_out=landmarks_out,
                history_len=history_maxlen,
            )
        case ModelType.CNN_RELU_LAST:
            return RegressionModelBatchnormFirst(
                roi_shape=roi_shape,
                landmarks_out=landmarks_out,
                history_len=history_maxlen,
            )
        case ModelType.CNN_MAX_POOL:
            return RegressionModelMaxPool(
                roi_shape=roi_shape,
                landmarks_out=landmarks_out,
                history_len=history_maxlen,
            )
        case ModelType.CNN_LSTM:  # TODO try different parameter inputs
            return CNNLSTM(
                num_classes=landmarks_out * 3, roi_shape=roi_shape, return_hidden_states=return_hidden_states
            )
        case ModelType.CNN_LSTM_EFFICIENT:  # TODO try different parameter inputs
            return EfficientCNNLSTM(
                num_classes=landmarks_out * 3, roi_shape=roi_shape, return_hidden_states=return_hidden_states
            )
