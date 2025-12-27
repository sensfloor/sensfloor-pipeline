import argparse
from pathlib import Path

import torch
import trackio
from torch.utils.data import DataLoader

from configs import HyperParams, data_root, get_hyper_param_configs, PROJECT_NAME
from data_loading.links_min_max import get_link_min_max
from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloorConfig
from data_loading.sensfloor_dataset import DatasetConfig, load_single_dataset, train_val_test_split
from model.pose_estimation_model import RegressionModel
from model.sensfloor_trainer import SensfloorTrainer, get_test_accuracy
from model.utils import set_seed
from utils import get_kept_links, get_device
from visualizations.create_landmark_predictions import create_predictions

PATCH_WIDTH = 4


def main(do_train: bool, do_test: bool) -> None:
    all_configs = get_hyper_param_configs()
    for hyper_params in all_configs:
        run_config(do_train, do_test, hyper_params)

def run_config(do_train: bool, do_test: bool, hyper_params: HyperParams) -> None:
    print(f"hyper params: {hyper_params}")

    set_seed(seed=hyper_params["seed"])

    device = get_device()

    trackio.init(
        project=PROJECT_NAME,
        config=dict(hyper_params),
        name=hyper_params["model_name"] # trackio checks for duplicate runs and changes the name in that case
        # space_id="JuliSharow/sensfloor", # Push to huggingface
    )

    drop_landmarks = hyper_params["dropped_landmarks"]
    kept_landmarks = [lm for lm in PoseLandmark if lm not in drop_landmarks]

    pose_to_model_index_dict = {landmark: i for i, landmark in enumerate(kept_landmarks)}

    floor_config = RoIFloorConfig(
        x_size=hyper_params["roi_x_size"],
        y_size=hyper_params["roi_y_size"],
        history_maxlen=hyper_params["roi_history_maxlen"],
        roi_size=hyper_params["roi_size"],
    )

    dataset_config = DatasetConfig(
        floor_config=floor_config,
        drop_landmarks=drop_landmarks,
        normalize_signals=hyper_params["do_normalize"],
        normalize_to_max=hyper_params["normalize_to_max"],
    )

    roi_shape = (dataset_config.floor_config.roi_size * PATCH_WIDTH, dataset_config.floor_config.roi_size * PATCH_WIDTH)
    landmarks_out = len(kept_landmarks)

    model_name = f"{hyper_params['model_name']}_model.pth"

    if do_train:
        model = RegressionModel(roi_shape=roi_shape, landmarks_out=landmarks_out, history_len=dataset_config.floor_config.history_maxlen)

        train_loader, val_loader, _ = train_val_test_split(
            data_root_path=data_root,
            ratios=hyper_params["split_ratios"],
            config=dataset_config,
            batch_size=hyper_params["batch_size"]
        )

        kept_links = get_kept_links(drop_landmarks)
        link_min, link_max = get_link_min_max(do_compute_link_lengths=True, links=kept_links)

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
            landmarks_out=landmarks_out,
            kept_links=kept_links,
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

        test_accuracy = get_test_accuracy(model, dataloader, landmarks_out)
        print(f"test_accuracy for {data_path} is :{test_accuracy}")
        trackio.log({"test_accuracy": test_accuracy})

    trackio.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Include flag if model should train")
    parser.add_argument("--test", action="store_true", help="Include flag if model should create test predictions")

    args = parser.parse_args()

    main(args.train, args.test)