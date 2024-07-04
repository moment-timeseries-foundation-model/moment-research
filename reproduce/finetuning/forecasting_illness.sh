#!/bin/bash

############################################
## Fine-tuning mode: Linear probing
############################################

### National_illness

# ### MOMENT
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 7\
#  --forecast_horizon 24\
#  --max_epoch 10\
#  --init_lr 0.0001\
#  --train_batch_size 16\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 7\
#  --forecast_horizon 60\
#  --max_epoch 10\
#  --init_lr 0.0001\
#  --train_batch_size 16\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'

# ### NBeats
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/nbeats.yaml'\
#  --gpu_id 4\
#  --forecast_horizon 24\
#  --max_epoch 10\
#  --init_lr 0.0001\
#  --train_batch_size 16\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/nbeats.yaml'\
#  --gpu_id 4\
#  --forecast_horizon 60\
#  --max_epoch 10\
#  --init_lr 0.0001\
#  --train_batch_size 16\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'

# ### GPT4TS
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 2\
#  --forecast_horizon 24\
#  --max_epoch 10\
#  --init_lr 0.0001\
#  --train_batch_size 16\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 2\
#  --forecast_horizon 60\
#  --max_epoch 10\
#  --init_lr 0.0001\
#  --train_batch_size 16\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'

# ### NHITS
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/nhits.yaml'\
#  --gpu_id 3\
#  --forecast_horizon 24\
#  --max_epoch 10\
#  --init_lr 0.0001\
#  --train_batch_size 16\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/nhits.yaml'\
#  --gpu_id 3\
#  --forecast_horizon 60\
#  --max_epoch 10\
#  --init_lr 0.0001\
#  --train_batch_size 16\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'
