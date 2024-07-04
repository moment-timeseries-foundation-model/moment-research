### Weather
python3 ../../scripts/baselines/gpt4ts_imputation.py\
 --config '../../configs/imputation/gpt4ts_train.yaml'\
 --gpu_id 3\
 --n_channels 21\
 --train_batch_size 32\
 --val_batch_size 128\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/weather.csv'
