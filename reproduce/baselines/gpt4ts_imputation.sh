#!/bin/bash

### ETTh1
python3 ../../scripts/baselines/gpt4ts_imputation.py\
 --config '../../configs/imputation/gpt4ts_train.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

### ETTh2
python3 ../../scripts/baselines/gpt4ts_imputation.py\
 --config '../../configs/imputation/gpt4ts_train.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv'

### ETTm1
python3 ../../scripts/baselines/gpt4ts_imputation.py\
 --config '../../configs/imputation/gpt4ts_train.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv'
