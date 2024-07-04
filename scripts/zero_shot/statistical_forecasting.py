import argparse
import os

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    Naive,
    RandomWalkWithDrift,
    SeasonalNaive,
)
from tqdm import trange

from moment.data.dataloader import get_timeseries_dataloader
from moment.data.forecasting_datasets import get_forecasting_datasets
from moment.utils.config import Config
from moment.utils.forecasting_metrics import get_forecasting_metrics
from moment.utils.utils import parse_config


def preprecess_dataset_for_statsforecast(dataset) -> pd.DataFrame:
    histories = []
    targets = []
    timestamps_history = []
    timestamps_target = []
    unique_ids_history = []
    unique_ids_target = []
    forecast_horizon = dataset.forecast_horizon

    for i in trange(dataset.length_dataset):
        metadata = dataset[i].metadata
        history = dataset.data.iloc[i, :].series_value.to_numpy()
        timestamps = np.arange(1, len(history) + 1)
        target = history[-forecast_horizon:]
        history = history[:-forecast_horizon]

        histories.append(history)
        targets.append(target)
        timestamps_history.append(timestamps[:-forecast_horizon])
        timestamps_target.append(timestamps[-forecast_horizon:])
        unique_ids_history.append(len(history) * [metadata["series_name"]])
        unique_ids_target.append(len(target) * [metadata["series_name"]])

    histories = np.concatenate(histories, axis=0)
    targets = np.concatenate(targets, axis=0)
    timestamps_history = np.concatenate(timestamps_history, axis=0)
    timestamps_target = np.concatenate(timestamps_target, axis=0)
    unique_ids_history = np.concatenate(unique_ids_history, axis=0)
    unique_ids_target = np.concatenate(unique_ids_target, axis=0)

    history_df = pd.DataFrame(
        {"unique_id": unique_ids_history, "ds": timestamps_history, "y": histories}
    )
    target_df = pd.DataFrame(
        {"unique_id": unique_ids_target, "ds": timestamps_target, "Target": targets}
    )

    return history_df, target_df


def get_test_dataloaders(args):
    args.data_split = "val"
    args.batch_size = args.val_batch_size
    val_dataloader = get_timeseries_dataloader(args=args)
    args.data_split = "test"
    args.batch_size = args.val_batch_size
    test_dataloader = get_timeseries_dataloader(args=args)
    return val_dataloader, test_dataloader


HORIZON_MAPPING = {
    "hourly": 48,
    "daily": 14,
    "weekly": 13,
    "monthly": 18,
    "quarterly": 8,
    "yearly": 6,
    "other": 8,
}
SEASONAL_MAPPING = {
    "yearly": 1,
    "quarterly": 4,
    "monthly": 12,
    "weekly": 1,
    "daily": 1,
    "hourly": 24,
    "other": 1,
}
FREQUENCY_MAPPING = {
    "yearly": "Y",
    "quarterly": "Q",
    "monthly": "M",
    "weekly": "W",
    "daily": "D",
    "hourly": "h",
    "other": "Q",
}
FREQUENCIES = {
    "m3": ["yearly", "quarterly", "monthly", "other"],
    "m4": ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"],
}
MODEL_NAMES = ["AutoARIMA", "AutoETS", "AutoTheta", "SeasonalNaive", "Naive", "RWD"]
file_format = "tsf"


def run_experiment(config_path: str = "../../configs/forecasting/zero_shot.yaml"):
    short_forecasting_datasets = get_forecasting_datasets(collection="monash")

    DEFAULT_CONFIG_PATH = "../../configs/default.yaml"
    BASE_PATH = "/".join(short_forecasting_datasets[0].split("/")[:-1])

    config = Config(
        config_file_path=config_path, default_config_file_path=DEFAULT_CONFIG_PATH
    ).parse()
    args = parse_config(config)

    results = []
    for dataset in ["m3", "m4"]:
        for frequency in FREQUENCIES[dataset]:
            print(f"Dataset: {dataset}, Frequency: {frequency}")
            args.full_file_path_and_name = os.path.join(
                BASE_PATH, f"{dataset}_{frequency}_dataset.{file_format}"
            )
            args.dataset_names = args.full_file_path_and_name
            args.forecast_horizon = HORIZON_MAPPING[frequency]
            args.season_length = SEASONAL_MAPPING[frequency]
            args.frequency = FREQUENCY_MAPPING[frequency]

            val_dataloader, test_dataloader = get_test_dataloaders(args)
            print(f"Forecast horizon: {test_dataloader.dataset.forecast_horizon}")
            print(
                f"Length Test: {test_dataloader.dataset.length_dataset + val_dataloader.dataset.length_dataset}"
            )

            val_history_df, val_target_df = preprecess_dataset_for_statsforecast(
                val_dataloader.dataset
            )
            test_history_df, test_target_df = preprecess_dataset_for_statsforecast(
                test_dataloader.dataset
            )
            history_df = pd.concat([val_history_df, test_history_df], axis=0)
            target_df = pd.concat([val_target_df, test_target_df], axis=0)

            assert (
                test_dataloader.dataset.length_dataset
                + val_dataloader.dataset.length_dataset
                == len(history_df.unique_id.unique())
            )

            # print(f"# of time-series: {len(history_df.unique_id.unique())}")

            models = [
                AutoARIMA(season_length=args.season_length),
                AutoETS(season_length=args.season_length),
                AutoTheta(season_length=args.season_length),
                SeasonalNaive(season_length=args.season_length),
                Naive(),
                RandomWalkWithDrift(),
            ]

            sf = StatsForecast(
                models=models, freq=args.frequency, n_jobs=args.n_jobs, verbose=True
            )
            sf.fit(history_df)

            forecast_df = sf.predict(h=args.forecast_horizon)
            forecast_df.reset_index(inplace=True)

            # Add the true valus to the dataframe
            forecast_df = forecast_df.merge(target_df, on=["unique_id", "ds"])

            assert (
                test_dataloader.dataset.length_dataset
                + val_dataloader.dataset.length_dataset
                == len(forecast_df.unique_id.unique())
            )

            for model_name in MODEL_NAMES:
                y_hat = forecast_df.loc[:, model_name]
                y = forecast_df.loc[:, "Target"]
                forecasting_metrics = get_forecasting_metrics(
                    y=y, y_hat=y_hat, reduction="mean"
                )
                results.append(
                    [
                        dataset,
                        frequency,
                        model_name,
                        forecasting_metrics.mape,
                        forecasting_metrics.smape,
                    ]
                )

            results_csv = pd.DataFrame(
                results, columns=["Dataset", "Frequency", "Model", "MAPE", "sMAPE"]
            )
            results_csv.to_csv(
                f"../../assets/results/zero_shot/statistical_forecasting_results.csv",
                index=False,
            )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="../../configs/forecasting/zero_shot.yaml"
    )

    args = parser.parse_args()

    run_experiment(config_path=args.config_path)
