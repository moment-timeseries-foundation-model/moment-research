#!/bin/bash

#############################################
### Fine-tuning mode: Linear probing
#############################################

### Weather

# ### MOMENT
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 0\
#  --forecast_horizon 96\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 0\
#  --forecast_horizon 192\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 0\
#  --forecast_horizon 336\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing_2.yaml'\
#  --gpu_id 0\
#  --forecast_horizon 720\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

# ### NBeats
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/nbeats.yaml'\
#  --gpu_id 4\
#  --forecast_horizon 96\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/nbeats.yaml'\
#  --gpu_id 4\
#  --forecast_horizon 720\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

# ### GPT4TS
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 3\
#  --forecast_horizon 96\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 3\
#  --forecast_horizon 720\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

### NHITS
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/nhits.yaml'\
#  --gpu_id 2\
#  --forecast_horizon 96\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/nhits.yaml'\
#  --gpu_id 2\
#  --forecast_horizon 720\
#  --train_batch_size 32\
#  --val_batch_size 128\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'
