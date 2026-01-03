import csv
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

from data_collection.mediapipe_utils import HEADER
from data_loading.pose_landmark import PoseLandmark
from data_loading.sensfloor_dataset import load_single_dataset, \
    DatasetConfig
from model.pose_estimation_model import RegressionModel

def detailed_collate_fn(batch: list):
    # 'batch' is a list of DetailedSensfloorPosesData objects

    # 1. Extract and stack tensors for the model (Creates B x C x H x W)
    #    (Assumes your object has 'transformed_roi_tensor')
    tensors = torch.stack([item.transformed_roi_tensor for item in batch])

    # 2. Keep the original objects for metadata access
    detailed_objects = batch

    return tensors, detailed_objects


def create_predictions(data_path: Path, dataset_config: DatasetConfig,  kept_landmarks: list[PoseLandmark], model: RegressionModel,
                       out_path: Path, total_mediapipe_landmarks: int = 33):
    detailed_dataset = load_single_dataset(data_path, config=dataset_config, return_detailed=True)
    detailed_dataloader = DataLoader(detailed_dataset, batch_size=1, shuffle=False, collate_fn=detailed_collate_fn)

    with open(out_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(HEADER)

        for batch_tensors, batch_details in tqdm.tqdm(detailed_dataloader):

            detailed_data = batch_details[0] #assuming the loader is batch_size 1
            outputs = model(batch_tensors)

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

            writer.writerow([detailed_data.frame_number] + full_row_data)
