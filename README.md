# SensFloor Pipeline

This repository is part of a project module submission. The overall goal is to estimate a 3D human pose from SensFloor signals.  
This repo contributes by providing utilities for:
1) Collecting a dataset for training a pose estimation model 
2) Training of a pose estimation model
3) Live-inference of the model by reading SensFloor signals, predicting a pose and sending it via a WebSocket to a client
4) Visualizations to analyze the data

All entry points are located in [`scripts/`](./scripts/). If you use VS Code, you can also look into [`launch.json`](.vscode/launch.json) where run configurations for the scripts are defined.


## Installation
Before executing any of the scripts install all the required packages by executing this command:  
`conda env create -f environment.yml`

## Data collection
We record SensFloor signals and video simultaneously to extract target poses via MediaPipe.
1) **Record Data** (Requires a connected camera and connection to SensFloor):  
  `python -m scripts.data_collection.collect_sensfloor_data_and_video --serial-port /dev/tty.usbserial-A571V1ZF`
3) **Extract poses** (Use [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=de) to extract target poses for training):  
`python -m scripts.data_collection.extract_mediapipe_poses --data data/train`

The dataset we collected for training can be downloaded [here](https://drive.google.com/file/d/1644e4UxIEFCMxIj_lLqDMzD8uOqHAij1/view?usp=sharing).

## Training
Before using the dataset we first define a configuration containing all required hyperparameters for training. Our final model uses the `_BASE_CONFIG` found in [`src/training/configs.py`](src/training/configs.py). To start training run this command:   
```python -m scripts.training.train```

### Post-training evaluation
After training a model these scripts can be used to evaluate its performance:
 | Description                                | Command                                                                                         |
 | ------------------------------------------ | ----------------------------------------------------------------------------------------------- |
 | Print human-readable metrics in terminal   | `python -m scripts.training.print_metrics --model outputs/models/sensfloor_model`               |
 | Evaluate model performance on the test set | `python -m scripts.training.create_test_metrics --model outputs/models/sensfloor_model`         |
 | Generate CSV predictions for hold-out data | `python -m scripts.training.create_hold_out_predictions --model outputs/models/sensfloor_model` |

## Live-inference
Once trained, the model can be used to predict poses in real-time from SensFloor signals and stream them via WebSocket to ws://127.0.0.1:8765. To 

### 1. Starting Inference Server
Depending whether you have a SensFloor connection at hand, choose one of the following modes:
- **With connection**: `python -m scripts.inference.run_server --model outputs/models/sensfloor_model --serial-port /dev/tty.usbserial-A571V1ZF`
- **Without connection** (mock signals by using a prerecorded readout): `python -m scripts.inference.run_server --model outputs/models/sensfloor_model --mock-file data/extra/2025-12-09_15-52-01-line-subject4/sensfloor_readout.csv`

### 2. Connect and Visualize
To verify that the stream is working, you have two options:
- **Terminal output** (runs a simple client on the terminal that prints the received poses for debugging purposes): `python -m scripts.inference.run_test_client`
- **3D visualization**: For a visual representation of the predicted poses, use our dedicated [frontend](https://github.com/sensfloor/sensfloor-frontend)


## Visualizations
While developing this project, we've developed several visualizations to get a deeper understanding of the data and the model.

### Data Visualizations
 | Description                                                          | Command                                                                                                                                                                |
 | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
 | Side-by-side view of video and sensor signals                        | `python -m scripts.visualization.visualize_collected_data --data data/train/2025-12-16_12-20-54-line-subject1`                                                         |
 | Activity heatmaps showing total sensor activation                    | `python -m scripts.visualization.visualize_readouts --csv data/train/2025-12-16_12-20-54-line-subject1/sensfloor_readout.csv`                                          |
 | Visualization of extracted ROI and their frames                      | `python -m scripts.visualization.visualize_roi_floor --data data/train/2025-12-16_12-20-54-line-subject1 --data-index 120`                                             |
 | Boxplot of bone length deviations for evaluating MediaPope estimates | `python -m scripts.visualization.visualize_bone_lengths --csv data/train/2025-12-16_12-20-54-line-subject1/video_poses.csv`                                            |
 | Boxplot of the joint movement range to assess estimation difficulty  | `python -m scripts.visualization.visualize_joint_movement`                                                                                                             |
 | Visualization of the trajectory of a person walking on the floor     | `python -m scripts.visualization.visualize_position_tracking --data data/train/2025-12-16_12-20-54-line-subject1`                                                      |
 | Comparison of estimated vs target pose                               | `python -m scripts.visualization.visualize_pose_comparison --model outputs/models/sensfloor_model --data data/train/2025-12-16_12-20-54-line-subject1 --data-index -4` |
 | Comparison between raw and filtered pose estimates                   | `python -m scripts.visualization.visualize_kalman_filter --csv data/tracking/2026-02-09_09-35-00-house-subject3/sensfloor_readout.csv --start 10 --end 42`             |
 | Boxplot of the Mean Per Joint Position Error                         | `python -m scripts.visualization.visualize_metrics --model outputs/models/sensfloor_model`                                                                             |
 | Image of rotation transformation for data                            | `python -m scripts.visualization.visualize_rotation --data data/train/2025-12-16_12-20-54-line-subject1 --data-index 120`                                              |