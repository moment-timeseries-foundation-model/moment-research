### Electricity
python3 ../../scripts/baselines/timesnet_imputation.py\
 --config '../../configs/imputation/timesnet_train.yaml'\
 --gpu_id 0\
 --d_model 64\
 --d_ff 64\
 --n_channels 321\
 --train_batch_size 4\
 --val_batch_size 8\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/electricity.csv'

