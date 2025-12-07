import argparse

import torch
from torch.utils.data import DataLoader

from data_loading.load_data import train_val_test_split, DatasetConfig, load_single_recording
from data_loading.pose_landmark import PoseLandmark
from model.pose_estimation_model import RegressionModel
from model.sensfloor_trainer import SensfloorTrainer
from model.utils import set_seed

from pathlib import Path


def main(do_train: bool, do_test: bool) -> None:
    set_seed(seed=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model_path = Path("./model/best_model.pth")

    drop_landmarks = [PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE_OUTER,
                      PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE_OUTER,
                      PoseLandmark.MOUTH_RIGHT, PoseLandmark.MOUTH_LEFT, PoseLandmark.RIGHT_EAR, PoseLandmark.LEFT_EAR,
                      PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_INDEX, PoseLandmark.LEFT_PINKY,
                      PoseLandmark.RIGHT_INDEX, PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_PINKY]

    dataset_config = DatasetConfig(roi_size=3, floor_size_x=6, floor_size_y=4, history_maxlen=10,
                                   drop_landmarks=drop_landmarks)

    roi_shape = (dataset_config.roi_size * 4, dataset_config.roi_size * 4)
    landmarks_out = len(PoseLandmark) - len(drop_landmarks)

    if do_train:
        train_loader, val_loader, test_loader = train_val_test_split(ratios=(0.7, 0.10, 0.20), config=dataset_config)

        model = RegressionModel(roi_shape=roi_shape, landmarks_out=landmarks_out, history_len=dataset_config.history_maxlen)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        trainer = SensfloorTrainer(model=model, device=device, optimizer=optimizer, patience=5, use_early_stopping=True)
        trainer.train(train_loader=train_loader, validation_loader=val_loader, epochs=10)

    if do_test:

        dataset = load_single_recording(Path("./data/2025-12-02_12-30-55"), config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        model = torch.load(f=model_path)

        for roi_history, pose in dataloader:
            outputs = model(roi_history)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Include flag if model should train")
    parser.add_argument("--test", action="store_true", help="Include flag if model should create test predictions")

    args = parser.parse_args()

    main(args.train, args.test)
