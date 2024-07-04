#!/bin/bash

############################################
## Fine-tuning mode: Linear probing
############################################

### Weather
python3 scripts/finetuning/imputation.py\
 --finetuning_mode 'linear-probing'\
 --config 'configs/imputation/linear_probing.yaml'\
 --gpu_id 1\
 --init_lr 0.001\
 --max_epoch 5\
 --train_batch_size 32\
 --val_batch_size 128\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'
