import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

from moment.common import PATHS
from moment.data.load_data import convert_tsf_to_dataframe
from moment.utils.data import downsample_timeseries, upsample_timeseries

from .base import DataSplits, TaskDataset, TimeseriesData

warnings.filterwarnings("ignore")


DATA_COLLECTIONS = [
    "autoformer",
    "monash",
    "epidemic/preprocessed",
    "fred/preprocessed",
]
DATASETS_EPIDEMIC = ["EU-Flu", "ILI-US"]
DATASETS_EXTENSIONS = [".tsf", ".csv", ".npy"]


def get_forecasting_datasets(collection: str) -> list[str]:
    data_dir = os.path.join(PATHS.DATA_DIR, "forecasting", collection)
    datasets = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if any(f.endswith(ext) for ext in DATASETS_EXTENSIONS):
                if "meta" not in f:  # exclude dataset meta data
                    datasets.append(os.path.join(root, f))
    return datasets


def filter_epidemic_datasets(
    datasets_collection: list[str],
    dataset_name: str,
    indicator: str = "",  # Daily or Weekly for the EU-Flu dataset
) -> list[str]:
    """
    Filters a collection of dataset names based on specified epidemic dataset criteria.
    """
    if dataset_name not in DATASETS_EPIDEMIC:
        raise ValueError(f"dataset_name must be one of {DATASETS_EPIDEMIC}")
    if dataset_name == "ILI" and indicator:
        warnings.warn("indicator is not used for ILI datasets")
    return [d for d in datasets_collection if dataset_name in d and indicator in d]


class LongForecastingDataset(TaskDataset):
    def __init__(
        self,
        seq_len: int = 512,
        forecast_horizon: int = 96,
        full_file_path_and_name: str = "../TimeseriesDatasets/forecasting/autoformer/ETTh1.csv",
        data_split: str = "train",
        target_col: Optional[str] = "OT",
        scale: bool = True,
        data_stride_len: int = 1,
        task_name: str = "long-horizon-forecasting",
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        test_ratio: float = 0.3,
        output_type: str = "univariate",
        random_seed: int = 42,
        **kwargs,
    ):
        super(LongForecastingDataset, self).__init__()
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        forecast_horizon : int
            Length of the prediction sequence.
        full_file_path_and_name : str
            Name of the dataset.
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        target_col : str
            Name of the target column. 
            If None, the target column is the last column.
        scale : bool
            Whether to scale the dataset.
        data_stride_len : int
            Stride length when generating consecutive 
            time-series windows. 
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', 'pre-training', or , 'imputation'.
        train_ratio : float
            Ratio of the training set.
        val_ratio : float
            Ratio of the validation set.
        test_ratio : float
            Ratio of the test set.
        output_type : str
            The type of the output. One of 'univariate' 
            or 'multivariate'. If multivariate, either the 
            target column must be specified or the dataset
            is flattened along the channel dimension.
        random_seed : int
            Random seed for reproducibility.
        """
        # print('full_file_path_and_name', full_file_path_and_name)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = full_file_path_and_name

        self.dataset_name = full_file_path_and_name.split("/")[-1][:-4]

        self.data_split = data_split
        self.target_col = target_col
        self.scale = scale
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.output_type = output_type
        self.random_seed = random_seed

        # Input checking
        self._check_inputs()

        # Read data
        self._read_data()

    def _check_inputs(self):
        assert self.data_split in [
            "train",
            "test",
            "val",
        ], "data_split must be one of 'train', 'test' or 'val'"
        assert (
            self.task_name in ["long-horizon-forecasting", "pre-training", "imputation"]
        ), "task_name must be one of 'long-horizon-forecasting', 'pre-training', 'imputation'"
        assert self.output_type in [
            "univariate",
            "multivariate",
        ], "output_type must be one of 'univariate' or 'multivariate'"
        assert (
            self.train_ratio + self.val_ratio + self.test_ratio == 1
        ), "train_ratio + val_ratio + test_ratio must be equal to 1"
        assert self.data_stride_len > 0, "data_stride_len must be greater than 0"

    def __repr__(self):
        repr = (
            f"LongForecastingDataset(dataset_name={self.dataset_name},"
            + f"length_timeseries={self.length_timeseries},"
            + f"length_dataset={self.__len__()},"
            + f"n_channels={self.n_channels},"
            + f"seq_len={self.seq_len},"
            + f"forecast_horizon={self.forecast_horizon},"
            + f"data_split={self.data_split},"
            + f"target_col={self.target_col},"
            + f"scale={self.scale},"
            + f"data_stride_len={self.data_stride_len},"
            + f"task_name={self.task_name},"
            + f"train_ratio={self.train_ratio},"
            + f"val_ratio={self.val_ratio},"
            + f"test_ratio={self.test_ratio},"
            + f"output_type={self.output_type})"
        )
        return repr

    def _get_borders(self):
        ### This is for the AutoFormer datasets
        remaining_autoformer_datasets = [
            "electricity",
            "exchange_rate",
            "national_illness",
            "traffic",
            "weather",
        ]

        if "ETTm" in self.dataset_name:
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4

        elif "ETTh" in self.dataset_name:
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24

        elif self.dataset_name in remaining_autoformer_datasets:
            n_train = int(self.train_ratio * self.length_timeseries_original)
            n_test = int(self.test_ratio * self.length_timeseries_original)
            n_val = self.length_timeseries_original - n_train - n_test

        train_end = n_train
        val_start = train_end - self.seq_len
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        return DataSplits(
            train=slice(0, train_end),
            val=slice(val_start, val_end),
            test=slice(test_start, test_end),
        )

    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        # Following would only work for AutoFormer datasets
        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        if self.target_col in list(df.columns) and self.output_type == "univariate":
            df = df[[self.target_col]]
            self.n_channels = 1
        elif (
            self.target_col is None
            and self.output_type == "univariate"
            and self.n_channels > 1
        ):
            raise ValueError(
                "target_col must be specified if output_type\
                              is 'univariate' for multi-channel datasets"
            )

        data_splits = self._get_borders()

        if self.scale:
            train_data = df[data_splits.train]
            self.scaler.fit(train_data.values)
            df = self.scaler.transform(df.values)
        else:
            df = df.values

        if self.data_split == "train":
            self.data = df[data_splits.train, :]
        elif self.data_split == "val":
            self.data = df[data_splits.val, :]
        elif self.data_split == "test":
            self.data = df[data_splits.test, :]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "long-horizon-forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            return TimeseriesData(
                timeseries=self.data[seq_start:seq_end, :].T,
                forecast=self.data[seq_end:pred_end, :].T,
                input_mask=input_mask,
                name=self.dataset_name,
                metadata={
                    "target_col": self.target_col,
                    "output_type": self.output_type,
                },
            )

        elif self.task_name == "pre-training" or self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            return TimeseriesData(
                timeseries=self.data[seq_start:seq_end, :].T,
                input_mask=input_mask,
                name=self.dataset_name,
                metadata={
                    "target_col": self.target_col,
                    "output_type": self.output_type,
                },
            )

    def __len__(self):
        if self.task_name == "pre-training" or self.task_name == "imputation":
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "long-horizon-forecasting":
            return (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1

    def plot(self, idx, channel=0):
        timeseries_data = self.__getitem__(idx)
        forecast = timeseries_data.forecast[channel, :]
        timeseries = timeseries_data.timeseries[channel, :]
        # input_mask = timeseries_data.input_mask

        plt.title(f"idx={idx}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len),
            timeseries.flatten(),
            label="Time-series",
            c="darkblue",
        )
        if self.task_name == "long-horizon-forecasting":
            plt.plot(
                np.arange(self.seq_len, self.seq_len + self.forecast_horizon),
                forecast.flatten(),
                label="Forecast",
                c="red",
                linestyle="--",
            )

        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()

    def check_and_remove_nans(self):
        return NotImplementedError


class ShortForecastingDataset(TaskDataset):
    def __init__(
        self,
        seq_len: int = 512,
        full_file_path_and_name: str = "../TimeseriesDatasets/forecasting/monash/dominick_dataset.tsf",
        data_split: str = "train",
        scale: bool = True,
        task_name: str = "short-horizon-forecasting",
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        test_ratio: float = 0.3,
        random_seed: int = 42,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        downsampling_type="last",
        pad_mode="constant",
        pad_constant_values=0,
        **kwargs,
    ):
        super(ShortForecastingDataset, self).__init__()
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        full_file_path_and_name : str
            Name of the dataset.
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        scale : bool
            Whether to scale the dataset.
        task_name : str
            The task that the dataset is used for. One of
            'short-horizon-forecasting', 'pre-training', or 'imputation'.
        train_ratio : float
            Ratio of the training set.
        val_ratio : float
            Ratio of the validation set.
        test_ratio : float
            Ratio of the test set.
        random_seed : int
            Random seed for reproducibility.
        """

        self.seq_len = seq_len
        self.full_file_path_and_name = full_file_path_and_name

        self.dataset_name = full_file_path_and_name.split("/")[-1][:-4]
        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values

        self.n_channels = 1  # All these time-series are univariate

        # Input checking
        self._check_inputs()

        # Read data
        self._read_data()

    def _check_inputs(self):
        assert self.data_split in [
            "train",
            "test",
            "val",
        ], "data_split must be one of 'train', 'test' or 'val'"
        assert (
            self.task_name
            in ["short-horizon-forecasting", "pre-training", "imputation"]
        ), "task_name must be one of 'short-horizon-forecasting', 'pre-training', or 'imputation'"
        assert (
            self.train_ratio + self.val_ratio + self.test_ratio == 1
        ), "train_ratio + val_ratio + test_ratio must be equal to 1"

    def __repr__(self):
        repr = (
            f"ShortForecastingDataset(dataset_name={self.dataset_name},"
            + f"length_dataset={self.__len__()},"
            + f"seq_len={self.seq_len},"
            + f"forecast_horizon={self.forecast_horizon},"
            + f"data_split={self.data_split},"
            + f"scale={self.scale},"
            + f"task_name={self.task_name},"
            + f"n_channels={self.n_channels},"
            + f"train_ratio={self.train_ratio},"
            + f"val_ratio={self.val_ratio},"
            + f"test_ratio={self.test_ratio},"
        )
        return repr

    def _get_borders(self):
        n_train = int(self.train_ratio * self.length_dataset)
        n_test = int(self.test_ratio * self.length_dataset)
        n_val = self.length_dataset - n_train - n_test

        train_end = n_train
        val_start = train_end
        val_end = val_start + n_val
        test_start = val_end

        return DataSplits(
            train=slice(0, train_end),
            val=slice(val_start, val_end),
            test=slice(test_start, None),
        )

    def _split_long_timeseries(self, df):
        """Split singular long timeseries into multiple timeseries of length seq_len"""
        timeseries = np.asarray(df.iloc[0, 2])
        timeseries = timeseries[: len(timeseries) - len(timeseries) % self.seq_len]

        # Split timeseries into 512 length chunks
        chunks = np.array_split(
            timeseries, indices_or_sections=len(timeseries) / self.seq_len
        )
        chunks = np.asarray(chunks)

        df = pd.concat([df] * len(chunks), ignore_index=True)

        for i in range(len(chunks)):
            df.iat[i, 2] = chunks[i]
        df.series_name = [f"T{i}" for i in range(1, len(chunks) + 1)]

        return df

    def _list_of_long_timeseries(self, df):
        return [
            "solar_4_seconds_dataset",
            "wind_4_seconds_dataset",
            "saugeenday_dataset",
            "us_births_dataset",
            "sunspot_dataset_without_missing_values",
        ]

    def _read_data(self) -> TimeseriesData:
        if self.full_file_path_and_name.endswith(".tsf"):
            (
                df,
                frequency,
                forecast_horizon,
                contain_missing_values,
                contain_equal_length,
            ) = convert_tsf_to_dataframe(
                self.full_file_path_and_name,
                replace_missing_vals_with="NaN",
                value_column_name="series_value",
            )

        elif self.full_file_path_and_name.endswith(".npy"):
            frequency = None
            forecast_horizon = None
            contain_missing_values = False
            contain_equal_length = False

            HORIZON_MAPPING = {
                "hourly": 48,
                "daily": 14,
                "weekly": 13,
                "monthly": 18,
                "quarterly": 8,
                "yearly": 6,
            }
            for f in HORIZON_MAPPING:
                if f in self.full_file_path_and_name.lower():
                    frequency = f
                    forecast_horizon = HORIZON_MAPPING[f]
                    break
            if frequency is None:
                raise ValueError(
                    "Frequency not found in filename: {}".format(
                        self.full_file_path_and_name
                    )
                )

            data = np.load(self.full_file_path_and_name, allow_pickle=True)
            data = data[()]  # unpack the dictionary
            data = [(_id, series) for _id, series in data.items()]
            df = pd.DataFrame(data, columns=["series_name", "series_value"])

        else:
            raise ValueError(f"Unknown file type: {self.full_file_path_and_name}")

        if self.dataset_name in self._list_of_long_timeseries(df):
            df = self._split_long_timeseries(df)

        self.meta_data = {
            "frequency": frequency,
            "forecast_horizon": forecast_horizon,
            "contain_missing_values": contain_missing_values,
            "contain_equal_length": contain_equal_length,
        }

        self.forecast_horizon = forecast_horizon
        # NOTE: What should we do if the forecast_horizon
        if self.forecast_horizon is None:
            self.forecast_horizon = 8
        assert self.forecast_horizon > 0, "forecast_horizon must be greater than 0"

        self.length_dataset = df.shape[0]
        self.length_timeseries = df.shape[0]

        # Following line shuffles the dataset
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        data_splits = self._get_borders()

        if self.scale:
            df.series_value = df.series_value.apply(
                lambda i: (i - i.mean()) / (i.std(ddof=0) + 1e-7)
            )

        if self.data_split == "train":
            self.data = df.iloc[data_splits.train, :]
        elif self.data_split == "val":
            self.data = df.iloc[data_splits.val, :]
        elif self.data_split == "test":
            self.data = df.iloc[data_splits.test, :]

        self.length_dataset = self.data.shape[0]

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        data = self.data.iloc[index, :]
        metadata = self.meta_data
        metadata.update(
            {
                "series_name": data.series_name,
                # "start_timestamp": data.start_timestamp, # Some datasets might not have this
                "dataset_name": self.dataset_name,
            }
        )

        timeseries = np.asarray(data.series_value)
        assert timeseries.ndim == 1, "Time-series is not univariate"

        input_mask = np.ones(self.seq_len)

        if self.task_name == "short-horizon-forecasting":
            forecast = timeseries[-self.forecast_horizon :]
            timeseries = timeseries[: -self.forecast_horizon]
        else:
            forecast = None

        timeseries_len = len(timeseries)

        if timeseries_len <= self.seq_len:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )

        elif timeseries_len > self.seq_len:
            timeseries, input_mask = downsample_timeseries(
                timeseries, self.seq_len, sampling_type=self.downsampling_type
            )

        return TimeseriesData(
            timeseries=timeseries.reshape((self.n_channels, self.seq_len)),
            forecast=forecast.reshape((self.n_channels, self.forecast_horizon))
            if forecast is not None
            else None,
            input_mask=input_mask,
            name=self.dataset_name,
            metadata=metadata,
        )

    def plot(self, idx):
        timeseries_data = self.__getitem__(idx)
        forecast = timeseries_data.forecast
        timeseries = timeseries_data.timeseries

        plt.title(f"idx={idx}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len),
            timeseries.squeeze(),
            label="Time-series",
            c="darkblue",
        )
        if self.task_name == "short-horizon-forecasting":
            plt.plot(
                np.arange(
                    timeseries.shape[-1], timeseries.shape[-1] + self.forecast_horizon
                ),
                forecast.squeeze(),
                label="Forecast",
                c="red",
                linestyle="--",
            )

        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()


class EpidemicForecastingDataset(TaskDataset):
    def __init__(
        self,
        seq_len: int = 64,
        forecast_horizon: int = 8,
        full_file_path_and_name: str = "../TimeseriesDatasets/forecasting/epidemic/preprocessed/EU-Flu_Austria_Daily_ICU_occupancy.csv",
        data_split: str = "train",
        target_col: Optional[str] = "value",
        scale: bool = True,
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        test_ratio: float = 0.3,
        output_type: str = "univariate",
        random_seed: int = 42,
        **kwargs,
    ):
        super(EpidemicForecastingDataset, self).__init__()
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        forecast_horizon : int
            Length of the prediction sequence.
        full_file_path_and_name : str
            Name of the dataset.
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        target_col : str
            Name of the target column. 
            If None, the target column is the last column.
        scale : bool
            Whether to scale the dataset.
        data_stride_len : int
            Stride length when generating consecutive 
            time-series windows. 
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', 'pre-training', or 'imputation'.
        train_ratio : float
            Ratio of the training set.
        val_ratio : float
            Ratio of the validation set.
        test_ratio : float
            Ratio of the test set.
        output_type : str
            The type of the output. One of 'univariate' 
            or 'multivariate'. If multivariate, either the 
            target column must be specified or the dataset
            is flattened along the channel dimension.
        random_seed : int
            Random seed for reproducibility.
        """
        # print('full_file_path_and_name', full_file_path_and_name)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = full_file_path_and_name

        self.dataset_name = full_file_path_and_name.split("/")[-1][:-4]

        self.data_split = data_split
        self.target_col = target_col
        self.scale = scale
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.output_type = output_type
        self.random_seed = random_seed

        # Input checking
        self._check_inputs()

        # Read data
        self._read_data()

    def _check_inputs(self):
        assert self.data_split in [
            "train",
            "test",
            "val",
        ], "data_split must be one of 'train', 'test' or 'val'"
        assert self.task_name in [
            "forecasting",
            "pre-training",
            "imputation",
        ], "task_name must be one of 'forecasting', 'pre-training', or 'imputation'"
        assert self.output_type in [
            "univariate",
            "multivariate",
        ], "output_type must be one of 'univariate' or 'multivariate'"
        assert (
            self.train_ratio + self.val_ratio + self.test_ratio == 1
        ), "train_ratio + val_ratio + test_ratio must be equal to 1"
        assert self.data_stride_len > 0, "data_stride_len must be greater than 0"

    def __repr__(self):
        repr = (
            f"LongForecastingDataset(dataset_name={self.dataset_name},"
            + f"length_timeseries={self.length_timeseries},"
            + f"length_dataset={self.__len__()},"
            + f"n_channels={self.n_channels},"
            + f"seq_len={self.seq_len},"
            + f"forecast_horizon={self.forecast_horizon},"
            + f"data_split={self.data_split},"
            + f"target_col={self.target_col},"
            + f"scale={self.scale},"
            + f"data_stride_len={self.data_stride_len},"
            + f"task_name={self.task_name},"
            + f"train_ratio={self.train_ratio},"
            + f"val_ratio={self.val_ratio},"
            + f"test_ratio={self.test_ratio},"
            + f"output_type={self.output_type})"
        )
        return repr

    def _get_borders(self):
        n_train = int(self.train_ratio * self.length_timeseries)
        n_test = int(self.test_ratio * self.length_timeseries)
        n_val = self.length_timeseries - n_train - n_test

        train_end = n_train
        val_start = train_end - self.seq_len
        val_end = val_start + n_val
        test_start = val_end - self.seq_len

        return DataSplits(
            train=slice(0, train_end),
            val=slice(val_start, val_end),
            test=slice(test_start, None),
        )

    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries = df.shape[0]
        self.n_channels = df.shape[1] - 1

        if self.target_col in list(df.columns) and self.output_type == "univariate":
            df = df[[self.target_col]]
            self.n_channels = 1
        elif (
            self.target_col is None
            and self.output_type == "univariate"
            and self.n_channels > 1
        ):
            raise ValueError(
                "target_col must be specified if output_type\
                              is 'univariate' for multi-channel datasets"
            )

        data_splits = self._get_borders()

        if self.scale:
            self.scaler.fit(df.iloc[data_splits.train, :].values)
            df = self.scaler.transform(df.values)
        else:
            df = df.values

        if self.data_split == "train":
            self.data = df[data_splits.train, :]
        elif self.data_split == "val":
            self.data = df[data_splits.val, :]
        elif self.data_split == "test":
            self.data = df[data_splits.test, :]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "short-horizon-forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            return TimeseriesData(
                timeseries=self.data[seq_start:seq_end, :].reshape(
                    (self.n_channels, self.seq_len)
                ),
                forecast=self.data[seq_end:pred_end, :].reshape(
                    (self.n_channels, self.forecast_horizon)
                ),
                input_mask=input_mask,
                name=self.dataset_name,
                metadata={
                    "target_col": self.target_col,
                },
            )

        elif self.task_name == "pre-training" or self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            return TimeseriesData(
                timeseries=self.data[seq_start:seq_end, :].reshape(
                    (self.n_channels, self.seq_len)
                ),
                input_mask=input_mask,
                name=self.dataset_name,
                metadata={
                    "target_col": self.target_col,
                },
            )

    def __len__(self):
        return (self.length_timeseries - self.seq_len) // self.data_stride_len

    def plot(self, idx):
        timeseries_data = self.__getitem__(idx)
        forecast = timeseries_data.forecast
        timeseries = timeseries_data.timeseries
        # input_mask = timeseries_data.input_mask

        plt.title(f"idx={idx}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len),
            timeseries.flatten(),
            label="Time-series",
            c="darkblue",
        )
        if self.task_name == "short-horizon-forecasting":
            plt.plot(
                np.arange(self.seq_len, self.seq_len + self.forecast_horizon),
                forecast.flatten(),
                label="Forecast",
                c="red",
                linestyle="--",
            )

        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()

    def check_and_remove_nans(self):
        return NotImplementedError
