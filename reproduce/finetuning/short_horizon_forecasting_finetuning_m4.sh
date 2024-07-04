#!/bin/bash

#############################################
### Fine-tuning mode: Linear probing
#############################################

#############################################
### Source dataset M4
#############################################

# python3 ../../scripts/finetuning/short_horizon_forecasting.py\
#  --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
#  --gpu_id 0\
#  --train_batch_size 1024\
#  --val_batch_size 1024\
#  --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_yearly_dataset.tsf'

python3 ../../scripts/finetuning/short_horizon_forecasting.py\
 --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
 --gpu_id 0\
 --train_batch_size 128\
 --val_batch_size 256\
 --init_lr 0.002\
 --max_epoch 5\
 --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_quarterly_dataset.tsf'

python3 ../../scripts/finetuning/short_horizon_forecasting.py\
 --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
 --gpu_id 0\
 --train_batch_size 1024\
 --val_batch_size 1024\
 --init_lr 0.002\
 --max_epoch 10\
 --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_monthly_dataset.tsf'

# python3 ../../scripts/finetuning/short_horizon_forecasting.py\
#  --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
#  --gpu_id 0\
#  --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_weekly_dataset.tsf'

# python3 ../../scripts/finetuning/short_horizon_forecasting.py\
#  --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
#  --gpu_id 0\
#  --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_daily_dataset.tsf'