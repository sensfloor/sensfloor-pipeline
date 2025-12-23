import csv
from pathlib import Path

import tqdm
from torch.utils.data import DataLoader

from data_collection.mediapipe_utils import HEADER
from data_loading.pose_landmark import PoseLandmark
from model.pose_estimation_model import RegressionModel


def create_predictions(dataloader: DataLoader, kept_landmarks: list[PoseLandmark], model: RegressionModel,
                       out_path: Path, total_mediapipe_landmarks: int = 33):
    with open(out_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(HEADER)

        for frame, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            roi_history, pose = data
            outputs = model(roi_history)

            coords = outputs.view(-1, len(kept_landmarks), 3)

            full_row_data = [None] * (total_mediapipe_landmarks * 3)  # Structure: [x0, y0, z0, x1, y1, z1, ...]

            # Fill in only the landmarks the model predicted
            for model_index, lm_enum in enumerate(kept_landmarks):
                mp_index = lm_enum.value
                x, y, z = coords[0, model_index].tolist()

                base_idx = mp_index * 3
                full_row_data[base_idx] = x
                full_row_data[base_idx + 1] = y
                full_row_data[base_idx + 2] = z

            writer.writerow([frame] + full_row_data)
