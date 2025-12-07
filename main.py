import torch

from data_loading.load_data import train_val_test_split, DatasetConfig
from data_loading.pose_landmark import PoseLandmark
from model.pose_estimation_model import RegressionModel
from model.sensfloor_trainer import SensfloorTrainer
from model.utils import set_seed


def main() -> None:
    set_seed(seed=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)


    drop_landmarks = [PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE_OUTER,
                      PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE_OUTER,
                      PoseLandmark.MOUTH_RIGHT, PoseLandmark.MOUTH_LEFT, PoseLandmark.RIGHT_EAR, PoseLandmark.LEFT_EAR,
                      PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_INDEX, PoseLandmark.LEFT_PINKY,
                      PoseLandmark.RIGHT_INDEX, PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_PINKY]


    dataset_config = DatasetConfig(roi_size=3, floor_size_x=6, floor_size_y=4, history_maxlen=10,
                                   drop_landmarks=drop_landmarks)
    train_loader, val_loader, test_loader = train_val_test_split(ratios=(0.7, 0.10, 0.20), config=dataset_config)
    rois, poses = next(iter(train_loader))
    print(f"rois: {rois}")

    roi_shape = (dataset_config.roi_size * 4, dataset_config.roi_size * 4)
    landmarks_out = len(PoseLandmark) - len(drop_landmarks)
    model = RegressionModel(roi_shape=roi_shape, landmarks_out=landmarks_out, history_len=dataset_config.history_maxlen)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    trainer = SensfloorTrainer(model=model, device=device, optimizer=optimizer, patience=5, use_early_stopping=True)
    trainer.train(train_loader=train_loader, validation_loader=val_loader, epochs=10)


if __name__ == '__main__':
    main()
