from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent

TRAIN_DATA_PATH = ROOT_PATH / "data/train"
HOLD_OUT_DATA_PATH = ROOT_PATH / "data/hold_out"

MODELS_FOLDER_PATH = ROOT_PATH / "outputs/models"
