import argparse
import random
from pathlib import Path
from typing import TypedDict, Literal

import torch
import trackio
from torch.utils.data import DataLoader

from data_loading.links_min_max import get_link_min_max
from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloorConfig
from data_loading.sensfloor_dataset import DatasetConfig, load_single_dataset, train_val_test_split
from model.pose_estimation_model import RegressionModel
from model.sensfloor_trainer import SensfloorTrainer, get_test_accuracy
from model.utils import set_seed
from visualizations.create_landmark_predictions import create_predictions
from data_collection.mediapipe_pose_extraction import main as extract_poses

PATCH_WIDTH = 4


class Config(TypedDict):
    # Training Params
    epochs: int
    learning_rate: float
    batch_size: int
    seed: int
    split_ratios: tuple[float, float, float]

    # Model/Data Params
    patch_width: int
    roi_x_size: int
    roi_y_size: int
    roi_history_maxlen: int
    roi_size: int
    do_normalize: bool

    training_data_folders: list[str]
    dropped_landmarks: list[PoseLandmark]

    # Optimizer/Trainer Params
    scheduler_patience: int
    scheduler_min_lr: float
    scheduler_factor: float
    trainer_patience: int
    amplify_link_loss: int
    mse_loss: Literal["mean", "sum"]

    #Testing
    test_path: Path
    model_name: str


data_root = Path("./data")
training_folders = [f.name for f in data_root.iterdir() if f.is_dir()]

hyper_params: Config = {
    "epochs": 2,
    "learning_rate": 5.5e-5,
    "batch_size": 64,
    "seed": random.randint(0, 1_000_000), # used for model_name
    "split_ratios": (
        0.79,
        0.2,
        0.01,
    ),

    "patch_width": 4,
    "roi_x_size": 6,
    "roi_y_size": 4,
    "roi_history_maxlen": 10,
    "roi_size": 3,
    "do_normalize": False,

    "training_data_folders": training_folders,
    "dropped_landmarks": [],

    "scheduler_patience": 3,
    "scheduler_min_lr": 1e-6,
    "scheduler_factor": 0.1,
    "trainer_patience": 7,
    "amplify_link_loss": 1,
    "mse_loss": "mean",

    "test_path": Path("./data_testing/2025-12-09_15-58-52-line-justin"),
    "model_name": "test_model_name_hyper_param",

}
if len(hyper_params["model_name"]) == 0:
    hyper_params["model_name"] = f"{hyper_params['seed']}"


def main(do_train: bool, do_test: bool) -> None:

    set_seed(seed=hyper_params["seed"])
    extract_poses(date=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    trackio.init(
        project="sensfloor",
        config=dict(hyper_params),
        # space_id="JuliSharow/sensfloor", # Push to huggingface
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
    ] + hyper_params["dropped_landmarks"]
    kept_landmarks = [lm for lm in PoseLandmark if lm not in drop_landmarks]

    pose_to_model_index_dict = {landmark: i for i, landmark in enumerate(kept_landmarks)}

    floor_config = RoIFloorConfig(
        x_size=hyper_params["roi_x_size"],
        y_size=hyper_params["roi_y_size"],
        history_maxlen=hyper_params["roi_history_maxlen"],
        roi_size=hyper_params["roi_size"],
        do_normalize=hyper_params["do_normalize"],
    )

    dataset_config = DatasetConfig(
        floor_config=floor_config,
        drop_landmarks=drop_landmarks,
    )

    roi_shape = (dataset_config.floor_config.roi_size * PATCH_WIDTH, dataset_config.floor_config.roi_size * PATCH_WIDTH)
    landmarks_out = len(kept_landmarks)

    model_name = f"{hyper_params['model_name']}_model.pth"

    if do_train:
        model = RegressionModel(
            roi_shape=roi_shape,
            landmarks_out=landmarks_out,
            history_len=dataset_config.floor_config.history_maxlen,
        )

        train_loader, val_loader, _ = train_val_test_split(
            data_root_path=data_root,
            ratios=hyper_params["split_ratios"],
            config=dataset_config,
            batch_size=hyper_params["batch_size"]
        )

        link_min, link_max = get_link_min_max(do_compute_link_lengths=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=hyper_params["scheduler_patience"],
            min_lr=hyper_params["scheduler_min_lr"],
            factor=hyper_params["scheduler_factor"]
        )

        trainer = SensfloorTrainer(
            model=model,
            best_model_name=model_name,
            device=device,
            optimizer=optimizer,
            patience=hyper_params["trainer_patience"],
            use_early_stopping=True,
            link_min=link_min,
            link_max=link_max,
            pose_to_model_dict=pose_to_model_index_dict,
            amplify_link_loss=hyper_params["amplify_link_loss"],
            loss_reduction=hyper_params["mse_loss"],
            scheduler=scheduler,
        )

        trainer.train(train_loader=train_loader, validation_loader=val_loader, epochs=hyper_params["epochs"])

    if do_test:
        data_path = hyper_params["test_path"]
        dataset = load_single_dataset(data_path, config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        model_path = Path(model_name)
        model = RegressionModel(
            roi_shape=roi_shape,
            landmarks_out=landmarks_out,
            history_len=dataset_config.floor_config.history_maxlen,
        )
        checkpoint = torch.load(f=model_path)
        model.load_state_dict(state_dict=checkpoint)

        out_path = data_path / f"{hyper_params['model_name']}_predictions.csv"
        create_predictions(dataloader, kept_landmarks, model, out_path)

        test_accuracy = get_test_accuracy(model, dataloader)
        print(f"test_accuracy for {data_path} is :{test_accuracy}")
        trackio.log({"test_accuracy": test_accuracy})

    trackio.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Include flag if model should train")
    parser.add_argument("--test", action="store_true", help="Include flag if model should create test predictions")

    args = parser.parse_args()

    main(args.train, args.test)