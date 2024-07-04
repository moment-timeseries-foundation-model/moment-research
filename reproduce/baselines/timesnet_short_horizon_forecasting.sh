#!/bin/bash

#############################################
### Source dataset FRED
#############################################

python3 ../../scripts/baselines/timesnet_short_horizon_forecasting.py\
 --config '../../configs/forecasting/timesnet.yaml'\
 --gpu_id 0\
 --train_batch_size 1024\
 --val_batch_size 1024\
 --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_yearly_dataset.npy'

python3 ../../scripts/baselines/timesnet_short_horizon_forecasting.py\
 --config '../../configs/forecasting/timesnet.yaml'\
 --gpu_id 0\
 --train_batch_size 1024\
 --val_batch_size 1024\
 --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_quarterly_dataset.npy'

python3 ../../scripts/baselines/timesnet_short_horizon_forecasting.py\
 --config '../../configs/forecasting/timesnet.yaml'\
 --gpu_id 0\
 --train_batch_size 1024\
 --val_batch_size 1024\
 --dataset_names '/TimeseriesDatasets/forecasting/fred/preprocessed/fred_monthly_dataset.npy'

#############################################
### Source dataset M4
#############################################

python3 ../../scripts/baselines/timesnet_short_horizon_forecasting.py\
 --config '../../configs/forecasting/timesnet.yaml'\
 --gpu_id 0\
 --train_batch_size 1024\
 --val_batch_size 1024\
 --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_yearly_dataset.tsf'

python3 ../../scripts/baselines/timesnet_short_horizon_forecasting.py\
 --config '../../configs/forecasting/timesnet.yaml'\
 --gpu_id 0\
 --train_batch_size 1024\
 --val_batch_size 1024\
 --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_quarterly_dataset.tsf'

python3 ../../scripts/baselines/timesnet_short_horizon_forecasting.py\
 --config '../../configs/forecasting/timesnet.yaml'\
 --gpu_id 0\
 --train_batch_size 1024\
 --val_batch_size 1024\
 --dataset_names '/TimeseriesDatasets/forecasting/monash/m4_monthly_dataset.tsf'