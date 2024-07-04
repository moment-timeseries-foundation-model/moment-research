#!/bin/bash

DIRECTORY="/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/KDD21"
COMMAND=(
    python3 scripts/baselines/timesnet_anomaly_detection.py \
    --config configs/anomaly_detection/timesnet_train.yaml \
    --gpu_id 1 --dataset_names
)

for FILE in "$DIRECTORY"/*
do
    echo "${COMMAND[@]}" "$FILE"
    "${COMMAND[@]}" "$FILE"
done
