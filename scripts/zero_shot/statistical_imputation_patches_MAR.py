import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from moment.data.dataloader import get_timeseries_dataloader
from moment.utils.config import Config
from moment.utils.forecasting_metrics import get_forecasting_metrics
from moment.utils.masking import Masking
from moment.utils.utils import parse_config

imputation_datasets = [
    "/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv",
    "/TimeseriesDatasets/forecasting/autoformer/ETTm2.csv",
    "/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv",
    "/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv",
    "/TimeseriesDatasets/forecasting/autoformer/electricity.csv",
    "/TimeseriesDatasets/forecasting/autoformer/weather.csv",
]

mask_ratios = [0.125, 0.25, 0.375, 0.5]


def statistical_interpolation(y):
    y = pd.DataFrame(y)

    linear_y = y.interpolate(method="linear", axis=1).values
    nearest_y = y.interpolate(method="nearest", axis=1).values
    cubic_y = y.interpolate(method="cubic", axis=1).values

    return linear_y, nearest_y, cubic_y


def forward_backward_fill(y):
    return pd.DataFrame(y).ffill(axis=1).bfill(axis=1).values


def run_experiment(
    config_path: str = "../../configs/imputation/zero_shot.yaml",
    default_config_path: str = "../../configs/default.yaml",
):
    config = Config(
        config_file_path=config_path, default_config_file_path=default_config_path
    ).parse()
    # config['device'] = gpu_id if torch.cuda.is_available() else 'cpu'
    args = parse_config(config)

    args.output_type = "multivariate"
    args.seq_len = 512
    args.data_stride_len = 512
    args.batch_size = args.val_batch_size

    results = []
    for dataset_name in tqdm(imputation_datasets, total=len(imputation_datasets)):
        args.full_file_path_and_name = dataset_name
        args.dataset_names = args.full_file_path_and_name
        args.data_split = "test"
        test_dataloader = get_timeseries_dataloader(args=args)

        trues = []
        masks = {}
        mask_generators = {}
        for mask_ratio in mask_ratios:
            mask_generators[mask_ratio] = Masking(mask_ratio=mask_ratio)

        for batch_x in test_dataloader:
            timeseries = batch_x.timeseries.float()
            n_examples, n_channels, _ = timeseries.shape
            timeseries = timeseries.reshape((-1, 1, args.seq_len))

            input_mask = batch_x.input_mask.long()
            input_mask = input_mask.repeat_interleave(n_channels, axis=0)
            trues.append(timeseries.squeeze().numpy())

            for mask_ratio, mask_generator in mask_generators.items():
                if mask_ratio not in masks:
                    masks[mask_ratio] = []
                m = mask_generator.generate_mask(x=timeseries, input_mask=input_mask)
                assert m.shape == input_mask.shape, f"{m.shape} != {input_mask.shape}"
                masks[mask_ratio].append(m)

        trues = np.concatenate(trues, axis=0)
        for mask_ratio in mask_ratios:
            masks[mask_ratio] = np.concatenate(masks[mask_ratio], axis=0)

        for mask_ratio in tqdm(mask_ratios, total=len(mask_ratios)):
            preds = trues.copy()
            mask = masks[mask_ratio]
            preds[mask == 0] = torch.nan

            preds_fbfill = forward_backward_fill(preds.copy())
            preds_linear, preds_nearest, preds_cubic = statistical_interpolation(
                preds.copy()
            )

            metrics_fbfill = get_forecasting_metrics(
                y=trues[mask == 0], y_hat=preds_fbfill[mask == 0], reduction="mean"
            )
            metrics_linear = get_forecasting_metrics(
                y=trues[mask == 0], y_hat=preds_linear[mask == 0], reduction="mean"
            )
            metrics_nearest = get_forecasting_metrics(
                y=trues[mask == 0], y_hat=preds_nearest[mask == 0], reduction="mean"
            )
            metrics_cubic = get_forecasting_metrics(
                y=trues[mask == 0], y_hat=preds_cubic[mask == 0], reduction="mean"
            )

            results.append(
                [
                    dataset_name.split("/")[-1][:-4],
                    mask_ratio,
                    metrics_fbfill.mse,
                    metrics_fbfill.mae,
                    metrics_linear.mse,
                    metrics_linear.mae,
                    metrics_nearest.mse,
                    metrics_nearest.mae,
                    metrics_cubic.mse,
                    metrics_cubic.mae,
                ]
            )

    results = pd.DataFrame(
        results,
        columns=[
            "Dataset",
            "Mask Ratio",
            "FBfill (MSE)",
            "FBfill (MAE)",
            "Linear (MSE)",
            "Linear (MAE)",
            "Nearest (MSE)",
            "Nearest (MAE)",
            "Cubic (MSE)",
            "Cubic (MAE)",
        ],
    )
    results.to_csv(
        "../../assets/results/zero_shot/statistical_imputation_results_512_patches_MAR.csv",
        index=False,
    )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="../../configs/imputation/zero_shot.yaml"
    )

    args = parser.parse_args()

    run_experiment(config_path=args.config_path)
