### ETTh2
python3 scripts/finetuning/imputation.py\
 --finetuning_mode 'linear-probing'\
 --config 'configs/imputation/linear_probing.yaml'\
 --gpu_id 4\
 --init_lr 0.001\
 --max_epoch 5\
 --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv'
