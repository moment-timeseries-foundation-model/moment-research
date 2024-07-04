#!/bin/bash

#############################################
### Fine-tuning mode: Linear probing
#############################################

#############################################
### Source dataset FRED
#############################################

python3 ../../scripts/finetuning/short_horizon_forecasting.py\
 --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
 --gpu_id 3\
 --train_batch_size 128\
 --val_batch_size 256\
 --init_lr 0.002\
 --max_epoch 5\
 --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_quarterly_dataset.npy'

python3 ../../scripts/finetuning/short_horizon_forecasting.py\
 --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
 --gpu_id 3\
 --train_batch_size 1024\
 --val_batch_size 1024\
 --init_lr 0.002\
 --max_epoch 10\
 --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_monthly_dataset.npy'

# python3 ../../scripts/finetuning/short_horizon_forecasting.py\
#  --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
#  --gpu_id 3\
#  --train_batch_size 1024\
#  --val_batch_size 1024\
#  --init_lr 0.001\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_yearly_dataset.npy'

# python3 ../../scripts/finetuning/short_horizon_forecasting.py\
#  --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
#  --gpu_id 6\
#  --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_weekly_dataset.npy'

# python3 ../../scripts/finetuning/short_horizon_forecasting.py\
#  --config '../../configs/forecasting/linear_probing_short_horizon.yaml'\
#  --gpu_id 6\
#  --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_daily_dataset.npy'