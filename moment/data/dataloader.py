import logging

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.utils import shuffle
from torch.utils.data import ConcatDataset, DataLoader

from .anomaly_detection_datasets import (
    AnomalyDetectionDataset,
    get_anomaly_detection_datasets,
)
from .base import TimeseriesData
from .classification_datasets import ClassificationDataset, get_classification_datasets
from .forecasting_datasets import (
    LongForecastingDataset,
    ShortForecastingDataset,
    get_forecasting_datasets,
)


def _sample_datasets(dataset_name_to_class: dict, datasets_fraction: float = 1.0):
    if datasets_fraction < 1.0:
        end_idx = int(len(dataset_name_to_class) * datasets_fraction)
        shuffled_items = shuffle(list(dataset_name_to_class.items()))
        dataset_name_to_class = dict(shuffled_items[:end_idx])
    return dataset_name_to_class


def get_all_datasets(datasets_fraction: float = 1.0):
    classification_datasets = get_classification_datasets(collection="UCR")
    forecasting_datasets_long = get_forecasting_datasets(collection="autoformer")
    forecasting_datasets_short = get_forecasting_datasets(collection="monash")
    forecasting_datasets_short = forecasting_datasets_short + get_forecasting_datasets(
        collection="fred/preprocessed"
    )
    anomaly_detection_datasets = get_anomaly_detection_datasets(
        collection="TSB-UAD-Public"
    )

    dataset_name_to_class = {}
    for dataset in classification_datasets:
        dataset_name_to_class[dataset] = ClassificationDataset
    for dataset in forecasting_datasets_long:
        dataset_name_to_class[dataset] = LongForecastingDataset
    for dataset in forecasting_datasets_short:
        dataset_name_to_class[dataset] = ShortForecastingDataset
    for dataset in anomaly_detection_datasets:
        dataset_name_to_class[dataset] = AnomalyDetectionDataset

    dataset_name_to_class = _sample_datasets(dataset_name_to_class, datasets_fraction)
    datasets = list(dataset_name_to_class.keys())
    return datasets, dataset_name_to_class


def _get_labels(examples):
    labels = [example.labels for example in examples]
    labels = np.asarray(labels)
    return labels


def _get_forecasts(examples):
    forecasts = [torch.from_numpy(example.forecast) for example in examples]
    forecasts = torch.stack(forecasts)
    return forecasts


def _collate_fn_basic(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples]
    input_masks = [torch.from_numpy(example.input_mask) for example in examples]
    names = [example.name for example in examples]
    timeseries = torch.stack(timeseries)
    input_masks = torch.stack(input_masks)
    names = np.asarray(names)

    return TimeseriesData(timeseries=timeseries, input_mask=input_masks, name=names)


def _collate_fn_classification(examples):
    batch = _collate_fn_basic(examples)
    batch.labels = _get_labels(examples)
    return batch


def _collate_fn_anomaly_detection(examples):
    batch = _collate_fn_basic(examples)
    batch.labels = _get_labels(examples)
    return batch


def _collate_fn_forecasting(examples):
    batch = _collate_fn_basic(examples)
    batch.forecast = _get_forecasts(examples)
    return batch


def get_timeseries_dataloader(args, **kwargs):
    all_datasets, dataset_name_to_class = get_all_datasets(args.datasets_fraction)
    logging.debug(
        "dataset_names",
        args.dataset_names,
        type(args.dataset_names),
        args.dataset_names == "all",
    )
    if args.dataset_names == "all":
        assert (
            args.task_name == "pre-training"
        ), "Only pre-training task supports all datasets"
        args.dataset_names = all_datasets

        def init_dataset(name, cls):
            args.full_file_path_and_name = name
            return cls(**vars(args))

        dataset_classes = []
        dataset_classes = Parallel(n_jobs=args.num_workers)(
            delayed(init_dataset)(name, cls)
            for name, cls in dataset_name_to_class.items()
        )
        dataset = ConcatDataset(
            [ds for ds in dataset_classes if ds.length_timeseries >= args.seq_len]
        )
        # dataset = ConcatDataset(dataset_classes)

    elif isinstance(args.dataset_names, str):
        args.full_file_path_and_name = args.dataset_names
        dataset = dataset_name_to_class[args.dataset_names](**vars(args))

    elif isinstance(args.dataset_names, list):
        assert (
            args.task_name == "pre-training"
        ), "Only pre-training task supports multiple datasets"
        dataset_classes = []
        dataset_classes = Parallel(n_jobs=args.num_workers)(
            delayed(dataset_name_to_class[name])(**vars(args))
            for name in args.dataset_names
        )
        dataset = ConcatDataset(
            [ds for ds in dataset_classes if ds.length_timeseries >= args.seq_len]
        )
    else:
        raise NotImplementedError

    collate_fn_map = {
        "pre-training": _collate_fn_basic,
        "imputation": _collate_fn_basic,
        "classification": _collate_fn_classification,
        "long-horizon-forecasting": _collate_fn_forecasting,
        "short-horizon-forecasting": _collate_fn_forecasting,
        "anomaly-detection": _collate_fn_anomaly_detection,
    }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn_map[args.task_name],
    )

    return dataloader
