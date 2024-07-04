from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy.typing as npt
import torch
from torch.utils.data import Dataset


@dataclass
class TimeseriesData:
    timeseries: npt.NDArray = None
    forecast: npt.NDArray = None
    labels: Union[npt.NDArray, int, str] = None
    input_mask: npt.NDArray = None
    metadata: dict = None
    name: str = None


@dataclass
class TimeseriesOutputs:
    forecast: npt.NDArray = None
    anomaly_scores: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False


@dataclass
class DataSplits:
    train: npt.NDArray = None
    val: npt.NDArray = None
    test: npt.NDArray = None


@dataclass
class ClassificationResults:
    train_embeddings: npt.NDArray = None
    test_embeddings: npt.NDArray = None
    train_labels: npt.NDArray = None
    test_labels: npt.NDArray = None
    train_predictions: npt.NDArray = None
    test_predictions: npt.NDArray = None
    train_accuracy: float = None
    test_accuracy: float = None
    dataset_name: str = None


@dataclass
class AnomalyDetectionResults:
    dataset_name: str = None
    anomaly_scores: npt.NDArray = None
    labels: npt.NDArray = None
    observations: npt.NDArray = None
    predictions: npt.NDArray = None
    adjbestf1: float = None
    raucroc: float = None
    raucpr: float = None
    vusroc: float = None
    vuspr: float = None


class FPTDataset:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._ind = 0

    def get_batch(self, batch_size, train=True):
        x, y = self.get_batch_np(batch_size, train=train)
        x = torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
        y = torch.from_numpy(y).to(device=self.device, dtype=torch.long)
        self._ind += 1
        return x, y

    def get_batch_np(self, batch_size, train):
        raise NotImplementedError

    def start_epoch(self):
        self._ind = 0


class TaskDataset(ABC, Dataset):
    def __init__(self):
        super(TaskDataset, self).__init__()

    def _read_data(self) -> TimeseriesData:
        return NotImplementedError

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        return NotImplementedError

    def plot(self, idx):
        return NotImplementedError

    def _check_and_remove_nans(self):
        return NotImplementedError

    def _subsample(self):
        return NotImplementedError
