#!/bin/bash

############################################
## Fine-tuning mode: Linear probing
############################################

### ETTm1
python3 scripts/finetuning/imputation.py\
 --finetuning_mode 'linear-probing'\
 --config 'configs/imputation/linear_probing.yaml'\
 --gpu_id 0\
 --init_lr 0.001\
 --max_epoch 5\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv'

