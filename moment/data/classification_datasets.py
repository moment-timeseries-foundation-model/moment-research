import logging
import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from moment.common import PATHS
from moment.data.load_data import load_from_tsfile
from moment.utils.data import (
    downsample_timeseries,
    interpolate_timeseries,
    upsample_timeseries,
)

from .base import DataSplits, TaskDataset, TimeseriesData

warnings.filterwarnings("ignore")


# Taken from TS2Vec
DATASETS_WITHOUT_NORMALIZATION = [
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "BME",
    "Chinatown",
    "Crop",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "MelbournePedestrian",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "PowerCons",
    "Rock",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "SmoothSubspace",
    "UMD",
]


def get_classification_datasets(collection: str = "UCR", subset: Optional[str] = None):
    data_dir = os.path.join(PATHS.DATA_DIR, "classification", collection)
    datasets = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".ts") and "TRAIN" not in file:
                datasets.append(os.path.join(root, file))

    remove_datasets = ["CharacterTrajectories_TEST.ts"]
    datasets = [
        dataset for dataset in datasets if dataset.split("/")[-1] not in remove_datasets
    ]
    if subset == "univariate":
        summary = pd.read_csv("~/MOMENT/assets/data/summaryUnivariate.csv")
        univariate_datasets = summary.problem.tolist()
        datasets = [
            i for i in datasets if i.split("/")[-2].strip() in univariate_datasets
        ]
    return datasets


class ClassificationDataset(TaskDataset):
    def __init__(
        self,
        seq_len: int = 512,
        full_file_path_and_name: str = "../TimeseriesDatasets/classification/UCRArchieve_2018/ACSF1/ACSF1_TEST.ts",
        data_split: str = "train",
        scale: bool = True,
        task_name: str = "classification",
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        test_ratio: float = 0.3,
        output_type: str = "univariate",
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        downsampling_type="interpolate",
        pad_mode="constant",
        pad_constant_values=0,
        **kwargs,
    ):
        super(ClassificationDataset, self).__init__()
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        dataset_name : str
            Name of the dataset.
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        scale : bool
            Whether to scale the dataset.
        task_name : str
            The task that the dataset is used for. One of
            'classification' or 'pre-training'.
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
        """

        self.seq_len = seq_len
        self.full_file_path_and_name = full_file_path_and_name
        self.dataset_name = full_file_path_and_name.split("/")[-2]
        self.series = full_file_path_and_name.split("/")[-1].split("_")[0]

        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.output_type = output_type
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values

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
            "classification",
            "pre-training",
        ], "task_name must be one of 'classification' or 'pre-training'"
        assert self.output_type in [
            "univariate",
            "multivariate",
        ], "output_type must be one of 'univariate' or 'multivariate'"

    def __repr__(self):
        repr = (
            f"ClassificationDataset(dataset_name={self.dataset_name},"
            + f"n_timeseries={self.n_timeseries},"
            + f"dataset_size={self.__len__()},"
            + f"length_of_each_timeseries={self.length_timeseries},"
            + f"n_channels={self.n_channels},"
            + f"seq_len={self.seq_len},"
            + f"data_split={self.data_split},"
            + f"scale={self.scale},"
            + f"task_name={self.task_name},"
            + f"train_ratio={self.train_ratio},"
            + f"val_ratio={self.val_ratio},"
            + f"test_ratio={self.test_ratio},"
            + f"output_type={self.output_type})"
        )
        return repr

    def _get_borders(self):
        train_end = 0
        val_start = 0
        val_end = 0
        if self.data_split in ["train", "val"]:
            train_ratio, val_ratio = (
                self.train_ratio / (self.train_ratio + self.val_ratio),
                self.val_ratio / (self.train_ratio + self.val_ratio),
            )
            train_end = int(train_ratio * self.n_timeseries)

        return DataSplits(
            train=slice(0, train_end), val=slice(train_end, None), test=slice(0, None)
        )

    def _transform_labels(self, train_labels: np.ndarray, test_labels: np.ndarray):
        # Move the labels to {0, ..., L-1}
        labels = np.unique(train_labels)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        train_labels = np.vectorize(transform.get)(train_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, test_labels

    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        self.label_transformer_is_fitted = False

        root_path = self.full_file_path_and_name.split("/")[:-1]
        train_path = os.path.join("/".join(root_path), self.series + "_TRAIN.ts")
        _, train_labels = load_from_tsfile(train_path)

        if self.data_split == "train" or self.data_split == "val":
            path = os.path.join("/".join(root_path), self.series + "_TRAIN.ts")
        if self.data_split == "test":
            path = os.path.join("/".join(root_path), self.series + "_TEST.ts")

        self.data, self.labels = load_from_tsfile(path)
        _, self.labels = self._transform_labels(train_labels, self.labels)

        # Check if time-series have equal lengths. If not, left pad with zeros
        self._check_if_equal_length()

        # Check and remove NaNs
        self._check_and_remove_nans()

        if self.data.ndim == 3:
            self.n_timeseries, self.n_channels, self.length_timeseries = self.data.shape
        else:
            self.n_timeseries, self.length_timeseries = self.data.shape
            self.n_channels = 1

        if self.scale and self.dataset_name not in DATASETS_WITHOUT_NORMALIZATION:
            length_timeseries = self.data.shape[0]
            self.data = self.data.reshape(-1, self.data.shape[-1])
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)
            self.data = self.data.reshape(length_timeseries, -1, self.data.shape[-1])

        if self.output_type == "univariate":
            # Duplicate labels for multivariate datasets
            self.labels = np.repeat(
                self.labels.reshape(-1, 1), self.n_channels, axis=1
            ).reshape(-1, 1)
            self.data = self.data.reshape(-1, self.data.shape[-1])
            self.n_timeseries = self.data.shape[0]

        data_splits = self._get_borders()

        if self.data_split == "train":
            self.data = self.data[data_splits.train,]
            self.labels = self.labels[data_splits.train]
        elif self.data_split == "val":
            self.data = self.data[data_splits.val,]
            self.labels = self.labels[data_splits.val]
        elif self.data_split == "test":
            self.data = self.data[data_splits.test,]
            self.labels = self.labels[data_splits.test]

        self.n_timeseries = self.data.shape[0]
        self.data = self.data.T
        # self.input_mask = self.input_mask.T

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[:, index]
        timeseries_len = len(timeseries)
        labels = (
            self.labels[index,].astype(int)
            if self.task_name == "classification"
            else None
        )

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
            timeseries=np.expand_dims(timeseries, axis=0),
            labels=labels,
            input_mask=input_mask,
            name=self.dataset_name,
        )

    def __len__(self):
        return self.n_timeseries

    def _check_and_remove_nans(self):
        if np.isnan(self.data).any():
            logging.info("NaNs detected. Imputing values...")
            self.data = interpolate_timeseries(
                timeseries=self.data, interp_length=self.data.shape[-1]
            )
            self.data = np.nan_to_num(self.data)

    def _check_if_equal_length(self):
        if isinstance(self.data, list):
            n_timeseries = len(self.data)
            self.n_channels = self.data[0].shape[0]
            # Assume all time-series have the same number of channels
            # Then we have time-series of unequal lengths
            max_len = max([ts.shape[-1] for ts in self.data])
            for i, ts in enumerate(self.data):
                self.data[i] = interpolate_timeseries(
                    timeseries=ts, interp_length=max_len
                )
            self.data = np.asarray(self.data)
            logging.info(
                f"Time-series have unequal lengths. Reshaping to {self.data.shape}"
            )

    def plot(self, idx):
        timeseries_data = self.__getitem__(idx)
        label = timeseries_data.labels
        timeseries = timeseries_data.timeseries

        plt.title(f"idx={idx}, label={label}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len),
            timeseries.squeeze(),
            label="Time-series",
            c="darkblue",
        )
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()
