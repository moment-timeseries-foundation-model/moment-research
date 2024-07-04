#!/bin/bash

############################################
## Fine-tuning mode: Linear probing
############################################

### Electricity
python3 ../../scripts/finetuning/imputation.py\
 --finetuning_mode 'linear-probing'\
 --config '../../configs/imputation/linear_probing.yaml'\
 --gpu_id 2\
 --init_lr 0.001\
 --max_epoch 5\
 --train_batch_size 4\
 --val_batch_size 8\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/electricity.csv'


