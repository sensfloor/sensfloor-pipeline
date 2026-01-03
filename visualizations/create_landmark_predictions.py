import csv
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

from data_collection.mediapipe_utils import HEADER
from data_loading.pose_landmark import PoseLandmark
from data_loading.sensfloor_dataset import load_single_dataset, \
    DatasetConfig, DetailedSensfloorPosesData
from model.pose_estimation_model import RegressionModel
from model.sensfloor_trainer import SensfloorTrainer

ACC_HEADER = ["frame_number"] + [lm.name for lm in PoseLandmark]

def detailed_collate_fn(batch: list[DetailedSensfloorPosesData]):
    # 'batch' is a list of DetailedSensfloorPosesData objects

    # 1. Extract and stack tensors for the model (Creates B x C x H x W)
    #    (Assumes your object has 'transformed_roi_tensor')
    tensors = torch.stack([item.transformed_roi_tensor for item in batch])
    labels = torch.stack([item.transformed_label_tensor for item in batch])

    # 2. Keep the original objects for metadata access
    detailed_objects = batch

    return tensors, labels, detailed_objects


def create_predictions(data_path: Path,
                       dataset_config: DatasetConfig,
                       kept_landmarks: list[PoseLandmark],
                       model: RegressionModel,
                       pred_out_path: Path,
                       acc_out_path: Path, device,
                       total_mediapipe_landmarks: int = 33):
    detailed_dataset = load_single_dataset(data_path, config=dataset_config, return_detailed=True)
    detailed_dataloader = DataLoader(detailed_dataset, batch_size=256, shuffle=False, collate_fn=detailed_collate_fn)
    model.to(device)
    model.eval()


    with open(pred_out_path, "w", newline="") as f_pred, \
            open(acc_out_path, "w", newline="") as f_acc:

        pred_writer = csv.writer(f_pred)
        acc_writer = csv.writer(f_acc)

        pred_writer.writerow(HEADER)
        acc_writer.writerow(ACC_HEADER)

        # for each batch of epoch
        for batch_tensors, batch_labels, batch_details in tqdm.tqdm(detailed_dataloader):

            batch_tensors = batch_tensors.to(device)
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                outputs = model(batch_tensors)

                pred_coords = outputs.view(outputs.size(0), len(kept_landmarks), 3)
                accuarcy = SensfloorTrainer.calculate_joint_accuracies(outputs, batch_labels, landmarks_out=len(kept_landmarks))

                pred_cpu = pred_coords.cpu().numpy()
                accuarcy_cpu = accuarcy.cpu().numpy()

            # for each output of batch
            for coords, accuracy, detailed_data in zip(pred_cpu, accuarcy_cpu, batch_details):

                # The csv should have all 33 joints even though the model doesnt predict all of them
                full_pred_row = [None] * (total_mediapipe_landmarks * 3)
                full_acc_row = [None] * total_mediapipe_landmarks

                # insert kept landmarks into the full rows
                # for each joint of output
                for model_prediction_index, kept_landmark in tqdm.tqdm(enumerate(kept_landmarks)):
                    media_pipe_joint_index = kept_landmark.value # index of the mediapipe landmark
                    x, y, z = coords[model_prediction_index].tolist() # corresponding model predictions for landmark

                    base_idx = media_pipe_joint_index * 3
                    full_pred_row[base_idx] = x
                    full_pred_row[base_idx + 1] = y
                    full_pred_row[base_idx + 2] = z

                    full_acc_row[media_pipe_joint_index] = accuracy[model_prediction_index]

                pred_writer.writerow([detailed_data.frame_number] + full_pred_row)
                acc_writer.writerow( [detailed_data.frame_number] + full_acc_row)
