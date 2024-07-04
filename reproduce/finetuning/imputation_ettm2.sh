### ETTm2
python3 scripts/finetuning/imputation.py\
 --finetuning_mode 'linear-probing'\
 --config 'configs/imputation/linear_probing.yaml'\
 --gpu_id 3\
 --max_epoch 5\
 --init_lr 0.001\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv'