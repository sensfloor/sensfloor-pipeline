import argparse
from pathlib import Path
from typing import TypedDict

import torch
import trackio
from torch.utils.data import DataLoader

from data_loading.links_min_max import get_link_min_max
from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloorConfig
from data_loading.sensfloor_dataset import DatasetConfig, load_single_dataset, train_val_test_split
from model.pose_estimation_model import RegressionModel
from model.sensfloor_trainer import SensfloorTrainer
from model.utils import set_seed
from visualizations.create_landmark_predictions import create_predictions
from data_collection.mediapipe_pose_extraction import main as extract_poses

PATCH_WIDTH = 4


class Config(TypedDict):
    epochs: int
    learning_rate: float
    batch_size: int


def main(do_train: bool, do_test: bool) -> None:
    set_seed(seed=42)

    extract_poses(date=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    hyper_params: Config = {"epochs": 1, "learning_rate": 0.001, "batch_size": 64}
    trackio.init(
        project="sensfloor",
        config=dict(hyper_params),
        # space_id="JuliSharow/sensfloor",
    )

    drop_landmarks = [
        PoseLandmark.LEFT_EYE,
        PoseLandmark.LEFT_EYE_INNER,
        PoseLandmark.LEFT_EYE_OUTER,
        PoseLandmark.RIGHT_EYE,
        PoseLandmark.RIGHT_EYE_INNER,
        PoseLandmark.RIGHT_EYE_OUTER,
        PoseLandmark.MOUTH_RIGHT,
        PoseLandmark.MOUTH_LEFT,
        PoseLandmark.RIGHT_EAR,
        PoseLandmark.LEFT_EAR,
        PoseLandmark.LEFT_THUMB,
        PoseLandmark.LEFT_INDEX,
        PoseLandmark.LEFT_PINKY,
        PoseLandmark.RIGHT_INDEX,
        PoseLandmark.RIGHT_THUMB,
        PoseLandmark.RIGHT_PINKY,
    ]
    kept_landmarks = [lm for lm in PoseLandmark if lm not in drop_landmarks]

    pose_to_model_index_dict = {landmark: i for i, landmark in enumerate(kept_landmarks)}

    floor_config = RoIFloorConfig(x_size=6, y_size=4, history_maxlen=10, roi_size=3)
    dataset_config = DatasetConfig(
        floor_config=floor_config,
        drop_landmarks=drop_landmarks,
    )

    roi_shape = (dataset_config.floor_config.roi_size * PATCH_WIDTH, dataset_config.floor_config.roi_size * PATCH_WIDTH)
    landmarks_out = len(kept_landmarks)

    if do_train:
        model = RegressionModel(
            roi_shape=roi_shape,
            landmarks_out=landmarks_out,
            history_len=dataset_config.floor_config.history_maxlen,
        )
        train_loader, val_loader, _ = train_val_test_split(
            data_root_path=Path("./data"),
            ratios=(
                0.01,
                0.2,
                0.79,
            ),
            config=dataset_config,
        )

        link_min, link_max = get_link_min_max(do_compute_link_lengths=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3, min_lr=1e-6, factor=0.1)
        trainer = SensfloorTrainer(
            model=model,
            device=device,
            optimizer=optimizer,
            patience=7,
            use_early_stopping=True,
            link_min=link_min,
            link_max=link_max,
            pose_to_model_dict=pose_to_model_index_dict,
            amplify_link_loss=1,
            scheduler=scheduler,
        )

        trainer.train(train_loader=train_loader, validation_loader=val_loader, epochs=hyper_params["epochs"])

    if do_test:
        data_path = Path("./data_testing/2025-12-09_15-58-52-line-justin")
        dataset = load_single_dataset(Path(data_path), config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        model_path = Path("best_model_74.pth")
        model = RegressionModel(
            roi_shape=roi_shape,
            landmarks_out=landmarks_out,
            history_len=dataset_config.floor_config.history_maxlen,
        )
        checkpoint = torch.load(f=model_path)
        model.load_state_dict(state_dict=checkpoint)

        out_path = data_path / "predicted_poses.csv"
        create_predictions(dataloader, kept_landmarks, model, out_path)

    trackio.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Include flag if model should train")
    parser.add_argument("--test", action="store_true", help="Include flag if model should create test predictions")

    args = parser.parse_args()

    main(args.train, args.test)
