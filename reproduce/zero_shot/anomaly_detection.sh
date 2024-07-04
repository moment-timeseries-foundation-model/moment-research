#! /bin/bash

python3 scripts/zero_shot/anomaly_detection.py \
    --config_path configs/anomaly_detection/zero_shot.yaml \
    --run_name zero-shot-anomaly-detection \
    --pretraining_run_name charmed-bee-241 \
    --opt_steps 100000 \
    --gpu_id 0
