import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loading.load_data import train_val_test_split, DatasetConfig, load_single_recording
from data_loading.pose_landmark import PoseLandmark
from model.pose_estimation_model import RegressionModel
from model.sensfloor_trainer import SensfloorTrainer
from model.utils import set_seed
from visualizations.create_landmark_predictions import create_predictions

PATCH_WIDTH = 4


def main(do_train: bool, do_test: bool) -> None:
    set_seed(seed=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    drop_landmarks = [PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE_OUTER,
                      PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE_OUTER,
                      PoseLandmark.MOUTH_RIGHT, PoseLandmark.MOUTH_LEFT, PoseLandmark.RIGHT_EAR, PoseLandmark.LEFT_EAR,
                      PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_INDEX, PoseLandmark.LEFT_PINKY,
                      PoseLandmark.RIGHT_INDEX, PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_PINKY]
    kept_landmarks = [lm for lm in PoseLandmark if lm not in drop_landmarks]

    pose_to_model_index_dict = {}
    for i, landmark in enumerate(kept_landmarks):
        pose_to_model_index_dict[landmark] = i

    dataset_config = DatasetConfig(roi_size=3, floor_size_x=6, floor_size_y=4, history_maxlen=10,
                                   drop_landmarks=drop_landmarks)

    roi_shape = (dataset_config.roi_size * PATCH_WIDTH, dataset_config.roi_size * PATCH_WIDTH)
    landmarks_out = len(kept_landmarks)

    if do_train:
        model = RegressionModel(roi_shape=roi_shape, landmarks_out=landmarks_out,
                                history_len=dataset_config.history_maxlen)
        train_loader, val_loader, test_loader = train_val_test_split(ratios=(0.7, 0.10, 0.20), config=dataset_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        data_path = './data/2025-12-02_12-08-00/video_poses.csv'
        trainer = SensfloorTrainer(model=model, device=device, optimizer=optimizer, patience=5, use_early_stopping=True,
                                   links_path=data_path, pose_to_model_dict=pose_to_model_index_dict,
                                   amplify_link_loss=0)

        trainer.train(train_loader=train_loader, validation_loader=val_loader, epochs=1)

    if do_test:
        data_path = Path("./data/2025-12-02_12-30-55")
        dataset = load_single_recording(Path(data_path), config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        model_path = Path("./best_model.pth")
        model = RegressionModel(roi_shape=roi_shape, landmarks_out=landmarks_out,
                                history_len=dataset_config.history_maxlen)
        checkpoint = torch.load(f=model_path)
        model.load_state_dict(state_dict=checkpoint)

        out_path = data_path / "predicted_poses.csv"
        create_predictions(dataloader, kept_landmarks, model, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Include flag if model should train")
    parser.add_argument("--test", action="store_true", help="Include flag if model should create test predictions")

    args = parser.parse_args()

    main(args.train, args.test)
