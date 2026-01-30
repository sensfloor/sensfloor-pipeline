import torch
import trackio

from data_loading.floor import PATCH_SIZE
from training.dataset.load_data import DatasetType, train_val_test_split
from training.link_loss.links_min_max import get_link_min_max
from data_loading.pose_landmark import PoseLandmark
from data_loading.roi_floor import RoIFloorConfig
from definitions import HOLD_OUT_DATA_PATH, TRAIN_DATA_PATH
from training.configs import (
    CONFIG_FILE_NAME,
    MODELS_FOLDER_PATH,
    PROJECT_NAME,
    TrainingConfiguration,
)
from training.dataset.sensfloor_dataset import DatasetConfig
from training.trainer.sensfloor_trainer import SensfloorTrainer, get_test_accuracy
from training.utils import get_device, get_kept_links, get_model, set_seed
from visualization.create_landmark_predictions import create_predictions


def train(configuration: TrainingConfiguration) -> None:
    print(f"hyper params: {configuration}")

    model_folder = MODELS_FOLDER_PATH / configuration.model_name
    model_file_name = "best_model.pth"

    configuration.save(model_folder / CONFIG_FILE_NAME)
    set_seed(seed=configuration.seed)
    device = get_device()

    trackio.init(
        project=PROJECT_NAME,
        config=dict(configuration),
        name=configuration.model_name,  # trackio checks for duplicate runs and changes the name in that case
        # space_id="JuliSharow/sensfloor", # Push to huggingface
    )

    kept_landmarks = configuration.landmarks
    drop_landmarks = [lm for lm in PoseLandmark if lm not in kept_landmarks]

    pose_to_model_index_dict = {landmark: i for i, landmark in enumerate(kept_landmarks)}

    dataset_config = DatasetConfig(
        floor_config=RoIFloorConfig(
            x_size=configuration.floor_x_size,
            y_size=configuration.floor_y_size,
            history_maxlen=configuration.roi_history_maxlen,
            roi_size=configuration.roi_size,
            active_field_min_value=configuration.active_field_min_value,
            remove_noise=configuration.remove_noise
        ),
        drop_landmarks=drop_landmarks,
        normalize_signals=configuration.do_normalize,
        normalize_to_max=configuration.normalize_to_max,
        rotate_data=configuration.rotate_data,
    )

    if dataset_config.floor_config.roi_size is None:
        roi_shape = (
            dataset_config.floor_config.x_size * PATCH_SIZE,
            dataset_config.floor_config.y_size * PATCH_SIZE,
        )
    else:
        roi_shape = (
            dataset_config.floor_config.roi_size * PATCH_SIZE,
            dataset_config.floor_config.roi_size * PATCH_SIZE,
        )

    landmarks_out = len(kept_landmarks)

    train_loader, val_loader, test_loader = train_val_test_split(
        data_root_path=TRAIN_DATA_PATH,
        ratios=configuration.split_ratios,
        config=dataset_config,
        batch_size=configuration.batch_size,
        dataset_type = DatasetType.HISTORY
    )

    model = get_model(
        roi_shape,
        landmarks_out,
        dataset_config.floor_config.history_maxlen,
        configuration.model_type,
    )

    kept_links = get_kept_links(drop_landmarks)
    link_min, link_max = get_link_min_max(
        do_compute_link_lengths=True,
        links=kept_links,
        traing_folders=TRAIN_DATA_PATH,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=configuration.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        patience=configuration.scheduler_patience,
        min_lr=configuration.scheduler_min_lr,
        factor=configuration.scheduler_factor,
    )

    trainer = SensfloorTrainer(
        model=model,
        best_model_name=model_file_name,
        results_path=model_folder,
        device=device,
        optimizer=optimizer,
        patience=configuration.trainer_patience,
        use_early_stopping=True,
        landmarks_out=landmarks_out,
        kept_links=kept_links,
        link_min=link_min,
        link_max=link_max,
        pose_to_model_dict=pose_to_model_index_dict,
        amplify_link_loss=configuration.amplify_link_loss,
        scheduler=scheduler,
    )

    trainer.train(train_loader=train_loader, validation_loader=val_loader, epochs=configuration.epochs)

    test_metrics = get_test_accuracy(model, test_loader, device, trainer)
    metrics_str = " | ".join([f"{key.upper()}: {value:.4f}" for key, value in test_metrics.items()])

    print(metrics_str)
    trackio.log(test_metrics)
    trackio.finish()


def create_hold_out_predictions(configuration: TrainingConfiguration) -> None:
    print(f"hyper params: {configuration}")

    model_folder = MODELS_FOLDER_PATH / configuration.model_name
    model_file_name = "best_model.pth"
    model_path = model_folder / model_file_name

    device = get_device()

    kept_landmarks = configuration.landmarks
    drop_landmarks = [lm for lm in PoseLandmark if lm not in kept_landmarks]

    floor_config = RoIFloorConfig(
        x_size=configuration.floor_x_size,
        y_size=configuration.floor_y_size,
        history_maxlen=configuration.roi_history_maxlen,
        roi_size=configuration.roi_size,
        active_field_min_value=configuration.active_field_min_value,
        remove_noise=configuration.remove_noise,
        offset_strategy=configuration.roi_offset_strategy
    )

    dataset_config = DatasetConfig(
        floor_config=floor_config,
        drop_landmarks=drop_landmarks,
        normalize_signals=configuration.do_normalize,
        normalize_to_max=configuration.normalize_to_max,
        rotate_data=configuration.rotate_data,
    )

    if floor_config.roi_size is None:
        roi_shape = (
            floor_config.x_size * PATCH_SIZE,
            floor_config.y_size * PATCH_SIZE,
        )
    else:
        roi_shape = (
            floor_config.roi_size * PATCH_SIZE,
            floor_config.roi_size * PATCH_SIZE,
        )

    landmarks_out = len(kept_landmarks)

    hold_out_dirs = [HOLD_OUT_DATA_PATH / folder for folder in configuration.hold_out_data_folder]

    model = get_model(
        roi_shape,
        landmarks_out,
        dataset_config.floor_config.history_maxlen,
        configuration.model_type,
    )
    checkpoint = torch.load(f=model_path)
    model.load_state_dict(state_dict=checkpoint)

    # --- Create csv Predictions ---
    for directory in hold_out_dirs:
        preds = directory / f"{configuration.model_name}_predictions.csv"
        acc = directory / f"{configuration.model_name}_accuracies.csv"
        print(f"creating predictions: {preds}")
        create_predictions(
            directory,
            dataset_config,
            kept_landmarks,
            model,
            preds,
            acc,
            device,
            stop_after_x_batches=None,
        )
