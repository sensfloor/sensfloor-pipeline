import torch
import trackio
from torch.utils.data import DataLoader

from data_loading.links_min_max import get_link_min_max
from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloorConfig
from definitions import DATA_PATH, ROOT_PATH
from training.configs import ALL_CONFIGS, PROJECT_NAME, HyperParams, save_hyperparams
from training.sensfloor_dataset import DatasetConfig, load_single_dataset, train_val_test_split
from training.sensfloor_trainer import SensfloorTrainer, get_test_accuracy
from training.utils import get_device, get_kept_links, get_model, set_seed
from visualization.create_landmark_predictions import create_predictions

PATCH_WIDTH = 4
MODELS_FOLDER_PATH = ROOT_PATH / "outputs" / "models"


# TODO: Refactor to two sperate methods -> Train config, Test config
def run_config(do_train: bool, do_test: bool, hyper_params: HyperParams) -> None:
    print(f"hyper params: {hyper_params}")

    model_folder = MODELS_FOLDER_PATH / hyper_params["model_name"]
    model_file_name = "best_model.pth"
    model_path = model_folder / model_file_name

    save_hyperparams(hyper_params, model_folder)
    set_seed(seed=hyper_params["seed"])
    device = get_device()

    trackio.init(
        project=PROJECT_NAME,
        config=dict(hyper_params),
        name=hyper_params["model_name"],  # trackio checks for duplicate runs and changes the name in that case
        # space_id="JuliSharow/sensfloor", # Push to huggingface
    )

    kept_landmarks = hyper_params["kept_landmarks"]
    drop_landmarks = [lm for lm in PoseLandmark if lm not in kept_landmarks]

    pose_to_model_index_dict = {landmark: i for i, landmark in enumerate(kept_landmarks)}

    floor_config = RoIFloorConfig(
        x_size=hyper_params["floor_x_size"],
        y_size=hyper_params["floor_y_size"],
        history_maxlen=hyper_params["roi_history_maxlen"],
        roi_size=hyper_params["roi_size"],
    )

    dataset_config = DatasetConfig(
        floor_config=floor_config,
        drop_landmarks=drop_landmarks,
        normalize_signals=hyper_params["do_normalize"],
        normalize_to_max=hyper_params["normalize_to_max"],
        rotate_data=hyper_params["rotate_data"],
    )

    roi_shape = (dataset_config.floor_config.roi_size * PATCH_WIDTH, dataset_config.floor_config.roi_size * PATCH_WIDTH)
    landmarks_out = len(kept_landmarks)

    train_loader, val_loader, test_loader = train_val_test_split(
        data_root_path=DATA_PATH,
        ratios=hyper_params["split_ratios"],
        config=dataset_config,
        batch_size=hyper_params["batch_size"],
    )

    if do_train:
        model = get_model(roi_shape, landmarks_out, dataset_config.floor_config.history_maxlen, hyper_params["model_type"])


        kept_links = get_kept_links(drop_landmarks)
        link_min, link_max = get_link_min_max(do_compute_link_lengths=True, links=kept_links)

        optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=hyper_params["scheduler_patience"],
            min_lr=hyper_params["scheduler_min_lr"],
            factor=hyper_params["scheduler_factor"],
        )

        trainer = SensfloorTrainer(
            model=model,
            best_model_name=model_file_name,
            results_path=model_folder,
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
        data_path = hyper_params["create_predictions_path"]

        model = get_model(roi_shape, landmarks_out, dataset_config.floor_config.history_maxlen, hyper_params["model_type"])
        checkpoint = torch.load(f=model_path)
        model.load_state_dict(state_dict=checkpoint)

        # --- Create csv Predictions ---
        preds = data_path / f"{hyper_params['model_name']}_predictions.csv"
        acc = data_path / f"{hyper_params['model_name']}_accuracies.csv"
        print(f"creating predictions: {preds}")
        create_predictions(
            data_path,
            dataset_config,
            kept_landmarks,
            model,
            preds,
            acc,
            device,
            stop_after_x_batches=None,
        )

        # --- Test Accuracy ---
        test_accuracy = get_test_accuracy(model, test_loader, device, landmarks_out)
        print(f"test_accuracy is :{test_accuracy}")
        trackio.log({"test_accuracy": test_accuracy})

    trackio.finish()


if __name__ == "__main__":
    save_hyperparams(ALL_CONFIGS[0], ROOT_PATH)
