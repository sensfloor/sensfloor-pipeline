import random

import numpy as np
import torch

from data_loading.pose_landmark import LINKS

SIGNAL_Z = 18  # required for heatmap model, needs to be even


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
    elif torch.mps.is_available():
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
