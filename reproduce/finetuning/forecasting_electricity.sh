#!/bin/bash

#############################################
### Fine-tuning mode: Linear probing
#############################################

### Electricity

# ### MOMENT
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config configs/forecasting/linear_probing.yaml\
#  --gpu_id 5\
#  --forecast_horizon 96\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config configs/forecasting/linear_probing.yaml\
#  --gpu_id 5\
#  --forecast_horizon 192\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config configs/forecasting/linear_probing.yaml\
#  --gpu_id 4\
#  --forecast_horizon 336\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv
#  # --init_lr 0.0001\

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config configs/forecasting/linear_probing.yaml\
#  --gpu_id 4\
#  --forecast_horizon 720\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv
#  # --init_lr 0.0001\

# ### NBeats
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/nbeats.yaml'\
#  --gpu_id 4\
#  --forecast_horizon 96\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --max_epoch 10\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv
# # --init_lr 0.0001\

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/nbeats.yaml'\
#  --gpu_id 4\
#  --forecast_horizon 720\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --max_epoch 10\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv
# # --init_lr 0.0001\

# ### GPT4TS
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 6\
#  --forecast_horizon 96\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv
# # --init_lr 0.0001\

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 6\
#  --forecast_horizon 720\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv
# # --init_lr 0.0001\

# ### NHITS
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config configs/forecasting/nhits.yaml\
#  --gpu_id 3\
#  --forecast_horizon 96\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config configs/forecasting/nhits.yaml\
#  --gpu_id 3\
#  --forecast_horizon 720\
#  --train_batch_size 4\
#  --val_batch_size 8\
#  --dataset_names /TimeseriesDatasets/forecasting/autoformer/electricity.csv
