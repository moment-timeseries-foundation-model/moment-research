#!/bin/bash

############################################
## Fine-tuning mode: Linear probing
############################################

### ETTh1

### MOMENT
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 0\
#  --forecast_horizon 96\
#  --init_lr 0.0001\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 0\
#  --forecast_horizon 192\
#  --init_lr 0.0001\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 0\
#  --forecast_horizon 336\
#  --init_lr 0.0001\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 0\
#  --forecast_horizon 720\
#  --init_lr 0.0001\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

### NBeats
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config configs/forecasting/nbeats.yaml\
#  --gpu_id 4\
#  --forecast_horizon 96\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config configs/forecasting/nbeats.yaml\
#  --gpu_id 4\
#  --forecast_horizon 720\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# ### GPT4TS
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config configs/forecasting/gpt4ts.yaml\
#  --gpu_id 6\
#  --forecast_horizon 96\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config configs/forecasting/gpt4ts.yaml\
#  --gpu_id 6\
#  --forecast_horizon 720\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# ### TimesNet
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config configs/forecasting/timesnet.yaml\
#  --gpu_id 1\
#  --forecast_horizon 96\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config configs/forecasting/timesnet.yaml\
#  --gpu_id 1\
#  --forecast_horizon 720\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# ### N-HITS
# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/nhits.yaml'\
#  --gpu_id 3\
#  --forecast_horizon 96\
#  --init_lr 0.0001\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'

# python scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/nhits.yaml'\
#  --gpu_id 3\
#  --forecast_horizon 720\
#  --init_lr 0.0001\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv'
