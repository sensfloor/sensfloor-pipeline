import random

import numpy as np
import torch

SIGNAL_Z = 18 # required for heatmap model, needs to be even

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