import os
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from moment.common import PATHS

from .base import DataSplits, TaskDataset, TimeseriesData

warnings.filterwarnings("ignore")

DATA_COLLECTIONS = ["TSB-UAD-Artificial", "TSB-UAD-Public", "TSB-UAD-Synthetic"]


def get_anomaly_detection_datasets(collection: str = "TSB-UAD-Public"):
    data_dir = os.path.join(PATHS.DATA_DIR, "anomaly_detection", collection)
    datasets = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".out") and "train" not in file:
                datasets.append(os.path.join(root, file))

    return datasets


class AnomalyDetectionDataset(TaskDataset):
    def __init__(
        self,
        seq_len: int = 512,
        full_file_path_and_name: str = "../TimeseriesDatasets/anomaly_detection/NASA-MSL/C-1.train.out",
        data_split: str = "train",
        target_col: Optional[str] = None,
        scale: bool = True,
        data_stride_len: int = 1,
        task_name: str = "anomaly-detection",
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        test_ratio: float = 0.3,
        output_type: str = "univariate",
        random_seed: int = 42,
        downsampling_factor: Optional[int] = None,
        min_length: Optional[int] = None,
        **kwargs,
    ):
        super(AnomalyDetectionDataset, self).__init__()
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        dataset_name : str
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
            'anomaly-detection' or 'pre-training'.
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

        self.seq_len = seq_len
        self.full_file_path_and_name = full_file_path_and_name
        self.dataset_name = full_file_path_and_name.split("/")[-2]
        self.series = full_file_path_and_name.split("/")[-1].split(".")[0]

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

        # Downsampling for experiments. Refer to TSADAMS for more details
        self.downsampling_factor = downsampling_factor
        self.min_length = min_length  # Minimum length of time-series after downsampling for experiments

        # Input checking
        self._check_inputs()

        # Read data
        self._read_data()

    def _check_inputs(self):
        # Input checking
        assert self.data_split in [
            "train",
            "test",
            "val",
        ], "data_split must be one of 'train', 'test' or 'val'"
        assert self.task_name in [
            "anomaly-detection",
            "pre-training",
        ], "task_name must be one of 'anomaly-detection' or 'pre-training'"
        assert self.output_type in [
            "univariate",
            "multivariate",
        ], "output_type must be one of 'univariate' or 'multivariate'"
        assert self.data_stride_len > 0, "data_stride_len must be greater than 0"

    def __repr__(self):
        repr = (
            f"AnomalyDetectionDataset(dataset_name={self.dataset_name},"
            + f"length_timeseries={self.length_timeseries},"
            + f"length_dataset={self.__len__()},"
            + f"n_channels={self.n_channels},"
            + f"seq_len={self.seq_len},"
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

        # We use reconstruction based anomaly detection
        # so we do not need "context"
        train_end = n_train
        val_start = train_end  # - self.seq_len
        val_end = val_start + n_val
        test_start = val_end  # - self.seq_len

        return DataSplits(
            train=slice(0, train_end),
            val=slice(val_start, val_end),
            test=slice(test_start, -1),
        )

    def _get_borders_train_val(self):
        train_ratio = self.train_ratio / (self.train_ratio + self.val_ratio)
        n_train = int(train_ratio * self.length_timeseries)
        n_val = self.length_timeseries - n_train

        # We use reconstruction based anomaly detection
        # so we do not need "context"
        train_end = n_train
        val_start = train_end  # - self.seq_len
        val_end = val_start + n_val

        return DataSplits(train=slice(0, train_end), val=slice(val_start, val_end))

    def _get_borders_KDD21(self):
        train_ratio = self.train_ratio / (self.train_ratio + self.val_ratio)
        details = self.series.split("_")
        n_train = int(details[4])
        n_test = self.length_timeseries - n_train
        n_train = int(train_ratio * n_train)
        n_val = self.length_timeseries - n_train - n_test

        # We use reconstruction based anomaly detection
        # so we do not need "context"
        train_end = n_train
        val_start = train_end
        val_end = val_start + n_val
        test_start = val_end

        return DataSplits(
            train=slice(0, train_end),
            val=slice(val_start, val_end),
            test=slice(test_start, None),
        )

    def _read_and_process_NASA(self):
        def _load_one_split(data_split: str = "train"):
            data_split = "test" if data_split == "test" else "train"
            root_path = self.full_file_path_and_name.split("/")[:-1]
            path = os.path.join("/".join(root_path), self.series + f".{data_split}.out")
            df = pd.read_csv(path).infer_objects(copy=False).interpolate(method="cubic")
            return df.iloc[:, 0].values, df.iloc[:, -1].values.astype(int)

        self.n_channels = 1  # All TSB-UAD datasets are univariate
        timeseries, labels = _load_one_split(data_split=self.data_split)
        timeseries = timeseries.reshape(-1, 1)

        if self.scale:
            if self.data_split == "train":
                self.scaler.fit(timeseries)
            else:
                train_timeseries, _ = _load_one_split(data_split="train")
                train_timeseries = train_timeseries.reshape(-1, 1)
                self.scaler.fit(train_timeseries)
            timeseries = self.scaler.transform(timeseries).squeeze()

        self.length_timeseries = len(timeseries)

        data_splits = self._get_borders_train_val()

        # Normalize train and validation ratios
        if self.data_split == "train":
            data_splits = self._get_borders_train_val()
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits.train], labels[data_splits.train]
            )
        elif self.data_split == "val":
            data_splits = self._get_borders_train_val()
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits.val], labels[data_splits.val]
            )
        elif self.data_split == "test":
            self.data, self.labels = self._downsample_timeseries(timeseries, labels)

        self.length_timeseries = self.data.shape[0]

    def _downsample_timeseries(self, timeseries, labels):
        # Downsampling code taken from TSADAMS: https://github.com/mononitogoswami/tsad-model-selection/blob/src/tsadams/datasets/load.py#L100
        if (
            (self.downsampling_factor is not None)
            and (self.min_length is not None)
            and (len(timeseries) // self.downsampling_factor > self.min_length)
        ):
            padding = (
                self.downsampling_factor - len(timeseries) % self.downsampling_factor
            )
            timeseries = np.pad(timeseries, ((padding, 0)))
            labels = np.pad(labels, (padding, 0))

            timeseries = timeseries.reshape(
                timeseries.shape[-1] // self.downsampling_factor,
                self.downsampling_factor,
            ).max(axis=1)
            labels = labels.reshape(
                labels.shape[0] // self.downsampling_factor, self.downsampling_factor
            ).max(axis=1)

        return timeseries, labels

    def _read_and_process_KDD21(self):
        df = pd.read_csv(self.full_file_path_and_name)
        df.interpolate(inplace=True, method="cubic")

        self.length_timeseries = len(df)
        self.n_channels = 1
        labels = df.iloc[:, -1].values
        timeseries = df.iloc[:, 0].values.reshape(-1, 1)

        data_splits = self._get_borders_KDD21()

        if self.scale:
            self.scaler.fit(timeseries[data_splits.train])
            timeseries = self.scaler.transform(timeseries)
        timeseries = timeseries.squeeze()

        if self.data_split == "train":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits.train], labels[data_splits.train]
            )
        elif self.data_split == "val":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits.val], labels[data_splits.val]
            )
        elif self.data_split == "test":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits.test], labels[data_splits.test]
            )

        self.length_timeseries = self.data.shape[0]

    def _read_and_process_general(self):
        df = pd.read_csv(self.full_file_path_and_name)
        df.interpolate(inplace=True, method="cubic")

        self.length_timeseries = df.shape[0]
        self.n_channels = 1
        data_splits = self._get_borders()

        labels = df.iloc[:, -1].values
        timeseries = df.iloc[:, 0].values.reshape(-1, 1)

        if self.scale:
            self.scaler.fit(timeseries[data_splits.train, :])
            timeseries = self.scaler.transform(timeseries)
        timeseries = timeseries.squeeze()

        if self.data_split == "train":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits.train], labels[data_splits.train]
            )
        elif self.data_split == "val":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits.val], labels[data_splits.val]
            )
        elif self.data_split == "test":
            self.data, self.labels = self._downsample_timeseries(
                timeseries[data_splits.test], labels[data_splits.test]
            )

        self.length_timeseries = self.data.shape[0]

    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        if self.dataset_name in ["NASA-SMAP", "NASA-MSL"]:
            self._read_and_process_NASA()
        elif self.dataset_name in ["KDD21"]:
            self._read_and_process_KDD21()
        else:
            self._read_and_process_general()

    def __getitem__(self, index):
        if self.data.shape[-1] < self.seq_len:
            return None

        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "anomaly-detection":
            if seq_end > self.length_timeseries:
                seq_start = self.length_timeseries - self.seq_len
                seq_end = None

            return TimeseriesData(
                timeseries=self.data[seq_start:seq_end].reshape(
                    (self.n_channels, self.seq_len)
                ),
                labels=self.labels[seq_start:seq_end]
                .astype(int)
                .reshape((self.n_channels, self.seq_len)),
                input_mask=input_mask,
                name=self.dataset_name,
                metadata={
                    "target_col": self.target_col,
                },
            )

        elif self.task_name == "pre-training":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            return TimeseriesData(
                timeseries=self.data[seq_start:seq_end].reshape(
                    (self.n_channels, self.seq_len)
                ),
                input_mask=input_mask,
                name=self.dataset_name,
            )

    def __len__(self):
        return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1

    def _get_data_splits(self):
        return NotImplementedError

    def plot(self, idx):
        timeseries_data = self.__getitem__(idx)
        labels = timeseries_data.labels
        timeseries = timeseries_data.timeseries
        # input_mask = timeseries_data.input_mask

        plt.title(f"idx={idx}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len),
            timeseries.squeeze(),
            label="Time-series",
            c="darkblue",
        )

        if self.task_name == "anomaly-detection":
            plt.plot(
                np.arange(self.seq_len), labels, label="Labels", c="red", linestyle="--"
            )

        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()

    def check_and_remove_nans(self):
        return NotImplementedError
