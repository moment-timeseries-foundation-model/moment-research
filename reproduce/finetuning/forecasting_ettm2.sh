#!/bin/bash

#############################################
### Fine-tuning mode: Linear Probing
#############################################

### ETTm2

# ### MOMENT
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 6\
#  --forecast_horizon 96\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 6\
#  --forecast_horizon 192\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 6\
#  --forecast_horizon 336\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/linear_probing.yaml'\
#  --gpu_id 6\
#  --forecast_horizon 720\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# ### NBeats
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/nbeats.yaml'\
#  --gpu_id 4\
#  --forecast_horizon 96\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/nbeats.yaml'\
#  --gpu_id 4\
#  --forecast_horizon 720\
#  --max_epoch 10\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# ### GPT4TS
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 2\
#  --forecast_horizon 96\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'end-to-end'\
#  --config 'configs/forecasting/gpt4ts.yaml'\
#  --gpu_id 2\
#  --forecast_horizon 720\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# ### NHITS
# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/nhits.yaml'\
#  --gpu_id 6\
#  --forecast_horizon 96\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'

# python3 scripts/finetuning/forecasting.py\
#  --finetuning_mode 'linear-probing'\
#  --config 'configs/forecasting/nhits.yaml'\
#  --gpu_id 6\
#  --forecast_horizon 720\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'
