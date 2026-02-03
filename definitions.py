from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent

NEW_DATA_PATH = ROOT_PATH / "data/new"
VIDEO_FILENAME = "video.mp4"
READOUT_FILENAME = "sensfloor_readout.csv"

TRAIN_DATA_PATH = ROOT_PATH / "data/train"
HOLD_OUT_DATA_PATH = ROOT_PATH / "data/hold_out"

MODELS_FOLDER_PATH = ROOT_PATH / "outputs/models"
