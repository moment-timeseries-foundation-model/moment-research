### Electricity
python3 ../../scripts/baselines/gpt4ts_imputation.py\
 --config '../../configs/imputation/gpt4ts_train.yaml'\
 --gpu_id 1\
 --n_channels 321\
 --train_batch_size 4\
 --val_batch_size 8\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/electricity.csv'