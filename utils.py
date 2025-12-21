import torch

from data_loading.pose_landmark import LINKS


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