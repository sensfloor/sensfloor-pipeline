from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

NEW_DATA_PATH = ROOT_PATH / "data"
VIDEO_FILENAME = "video.mp4"
VIDEO_POSES_FILENAME = "video_poses.csv"
READOUT_FILENAME = "sensfloor_readout.csv"
TEST_METRICS_FILENAME = "test_metrics.csv"
BEST_MODEL_FILENAME = "best_model.pth"

TRAIN_DATA_PATH = ROOT_PATH / "data/train"
HOLD_OUT_DATA_PATH = ROOT_PATH / "data/hold-out"

MODELS_FOLDER_PATH = ROOT_PATH / "outputs/models"
