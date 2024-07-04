#!/bin/bash

# python3 scripts/zero_shot/anomaly_detection.py --experiment_name zero_shot_anomaly_detection --config_path ../../configs/anomaly_detection/zero_shot.yaml --gpu_id 0
python3 scripts/zero_shot/unsupervised_representation_learning.py --experiment_name unsupervised_representation_learning --config_path configs/classification/unsupervised_representation_learning.yaml --gpu_id 4