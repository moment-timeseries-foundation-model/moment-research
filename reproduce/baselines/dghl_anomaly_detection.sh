#!/bin/bash

DIRECTORY="/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/KDD21"
COMMAND=(
    python3 scripts/baselines/dghl_anomaly_detection.py \
    --config configs/anomaly_detection/dghl_train.yaml \
    --gpu_id 4 --dataset_names
)

for FILE in "$DIRECTORY"/*
do
    echo "${COMMAND[@]}" "$FILE"
    "${COMMAND[@]}" "$FILE"
done
