import pickle
from pathlib import Path

import torch
import trackio
from torch.utils.data import DataLoader

from data_loading.links_min_max import get_link_min_max
from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloorConfig
from definitions import ROOT_PATH, DATA_PATH
from training.configs import HyperParams, ModelType, PROJECT_NAME
from training.models.efficient_cnn import RegressionModelNoBatchnorm, RegressionModelMaxPool, RegressionReducedDim, \
    RegressionModelBatchnormFirst
from training.models.efficient_lstm import EfficientCNNLSTM
from training.models.lstm_model import CNNLSTM
from training.models.pose_estimation_model import RegressionModel
from training.sensfloor_dataset import DatasetConfig, train_val_test_split, load_single_dataset
from training.sensfloor_trainer import SensfloorTrainer, get_test_accuracy
from training.utils import get_device, get_kept_links
from training.utils import set_seed
from visualization.create_landmark_predictions import create_predictions

PATCH_WIDTH = 4
MODELS_FOLDER_PATH = ROOT_PATH / "outputs" / "models"
CONFIG_FILE_NAME = "config.pickle"


def save_config(config, model_folder: Path): # TODO Make jsonable
    model_folder.mkdir(parents=True, exist_ok=True)
    print(f"writing to file {model_folder}")
    with open(model_folder / CONFIG_FILE_NAME, "wb+") as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)


def load_config(model_folder: Path) -> HyperParams: # TODO Make jsonable
    with open(model_folder / CONFIG_FILE_NAME, "rb") as f:
        return pickle.load(f)


# TODO: Refactor to two sperate methods -> Train config, Test config
def run_config(do_train: bool, do_test: bool, hyper_params: HyperParams) -> None:
    print(f"hyper params: {hyper_params}")

    model_folder = MODELS_FOLDER_PATH / hyper_params['model_name']
    model_file_name = "best_model.pth"
    model_path = model_folder / model_file_name

    save_config(hyper_params, model_folder)
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
        rotate_data=hyper_params["rotate_data"],
    )

    roi_shape = (dataset_config.floor_config.roi_size * PATCH_WIDTH, dataset_config.floor_config.roi_size * PATCH_WIDTH)
    landmarks_out = len(kept_landmarks)

    def get_model(model_type: ModelType) -> torch.nn.Module:
        match model_type:
            case ModelType.CNN:
                return RegressionModel(
                    roi_shape=roi_shape,
                    landmarks_out=landmarks_out,
                    history_len=dataset_config.floor_config.history_maxlen,
                )
            case ModelType.CNN_EFFICIENT:
                return RegressionReducedDim(
                    roi_shape=roi_shape,
                    landmarks_out=landmarks_out,
                    history_len=dataset_config.floor_config.history_maxlen,
                )
            case ModelType.CNN_NO_BATCHNORM:
                return RegressionModelNoBatchnorm(
                    roi_shape=roi_shape,
                    landmarks_out=landmarks_out,
                    history_len=dataset_config.floor_config.history_maxlen,
                )
            case ModelType.CNN_RELU_LAST:
                return RegressionModelBatchnormFirst(
                    roi_shape=roi_shape,
                    landmarks_out=landmarks_out,
                    history_len=dataset_config.floor_config.history_maxlen,
                )
            case ModelType.CNN_MAX_POOL:
                return RegressionModelMaxPool(
                    roi_shape=roi_shape,
                    landmarks_out=landmarks_out,
                    history_len=dataset_config.floor_config.history_maxlen,
                )
            case ModelType.CNN_LSTM:  # TODO Assumes roi shape (12, 12)
                return CNNLSTM(num_classes=landmarks_out * 3)
            case ModelType.CNN_LSTM_EFFICIENT:  # TODO Assumes roi shape (12, 12)
                return EfficientCNNLSTM(num_classes=landmarks_out * 3)

    if do_train:
        model = get_model(hyper_params["model_type"])

        train_loader, val_loader, _ = train_val_test_split(
            data_root_path=DATA_PATH,
            ratios=hyper_params["split_ratios"],
            config=dataset_config,
            batch_size=hyper_params["batch_size"],
        )

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
        data_path = hyper_params["test_path"]

        model = get_model(hyper_params["model_type"])
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
        test_dataset = load_single_dataset(data_path, config=dataset_config)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_accuracy = get_test_accuracy(model, test_dataloader, device, landmarks_out)
        print(f"test_accuracy for {data_path} is :{test_accuracy}")
        trackio.log({"test_accuracy": test_accuracy})

    trackio.finish()
