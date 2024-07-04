#!/bin/bash

DIRECTORY="/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/KDD21"
COMMAND=(
    python3 scripts/baselines/gpt4ts_anomaly_detection.py \
    --config configs/anomaly_detection/gpt4ts_train.yaml \
    --gpu_id 1 --dataset_names
)

for FILE in "$DIRECTORY"/*
do
    echo "${COMMAND[@]}" "$FILE"
    "${COMMAND[@]}" "$FILE"
done
