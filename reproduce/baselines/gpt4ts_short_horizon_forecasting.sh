#!/bin/bash

#############################################
### Source dataset FRED
#############################################

# python3 ../../scripts/baselines/gpt4ts_short_horizon_forecasting.py\
#  --config '../../configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 1\
#  --train_batch_size 256\
#  --val_batch_size 512\
#  --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_yearly_dataset.npy'

# python3 ../../scripts/baselines/gpt4ts_short_horizon_forecasting.py\
#  --config '../../configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 3\
#  --train_batch_size 128\
#  --val_batch_size 256\
#  --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_quarterly_dataset.npy'

python3 ../../scripts/baselines/gpt4ts_short_horizon_forecasting.py\
 --config '../../configs/forecasting/gpt4ts.yaml'\
 --gpu_id 2\
 --train_batch_size 128\
 --val_batch_size 256\
 --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_monthly_dataset.npy'

#############################################
### Source dataset M4
#############################################

# python3 ../../scripts/baselines/gpt4ts_short_horizon_forecasting.py\
#  --config '../../configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 1\
#  --train_batch_size 256\
#  --val_batch_size 512\
#  --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_yearly_dataset.tsf'

# python3 ../../scripts/baselines/gpt4ts_short_horizon_forecasting.py\
#  --config '../../configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 3\
#  --train_batch_size 128\
#  --val_batch_size 256\
#  --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_quarterly_dataset.tsf'

python3 ../../scripts/baselines/gpt4ts_short_horizon_forecasting.py\
 --config '../../configs/forecasting/gpt4ts.yaml'\
 --gpu_id 2\
 --train_batch_size 128\
 --val_batch_size 256\
 --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_monthly_dataset.tsf'