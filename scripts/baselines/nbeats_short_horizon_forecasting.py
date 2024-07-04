import argparse
from typing import Optional

import torch

from moment.common import PATHS
from moment.tasks.forecast_finetune import ForecastFinetuning
from moment.utils.config import Config
from moment.utils.utils import control_randomness, make_dir_if_not_exists, parse_config

NOTES = "Train N-BEATS on source short-horizon forecasting datasets"

HORIZON_MAPPING = {
    "hourly": 48,
    "daily": 14,
    "weekly": 13,
    "monthly": 18,
    "other": 8,
    "quarterly": 8,
    "yearly": 6,
}


def forecasting(
    config_path: str = "../../configs/forecasting/nbeats.yaml",
    default_config_path: str = "../../configs/default.yaml",
    gpu_id: int = 0,
    train_batch_size: int = 64,
    val_batch_size: int = 256,
    finetuning_mode: str = "end-to-end",
    init_lr: Optional[float] = None,
    max_epoch: int = 10,
    dataset_names: str = "/TimeseriesDatasets/forecasting/monash/m3_monthly_dataset.tsf",
) -> None:
    config = Config(
        config_file_path=config_path, default_config_file_path=default_config_path
    ).parse()

    if isinstance(dataset_names, str):
        data_name = dataset_names.split("/")[-1].split(".")[0]
        frequency = data_name.split("_")[1]
        data_collection = data_name.split("_")[0]

    # Control randomness
    control_randomness(config["random_seed"])

    # Set-up parameters and defaults
    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    args = parse_config(config)
    make_dir_if_not_exists(config["checkpoint_path"])

    # Setup arguments
    args.source_dataset = f"{data_collection}_{frequency}"
    args.forecast_horizon = HORIZON_MAPPING[frequency]
    args.train_batch_size = train_batch_size
    args.val_batch_size = val_batch_size
    args.finetuning_mode = finetuning_mode
    args.max_epoch = max_epoch
    args.dataset_names = dataset_names
    if init_lr is not None:
        args.init_lr = init_lr

    print(f"Running experiments with config:\n{args}\n")
    task_obj = ForecastFinetuning(args=args)

    # Setup a W&B Logger
    task_obj.setup_logger(notes=NOTES)
    task_obj.train()

    # End the W&B Logger
    task_obj.end_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=256, help="Validation batch size"
    )
    parser.add_argument(
        "--init_lr", type=float, default=0.001, help="Peak learning rate"
    )
    parser.add_argument(
        "--finetuning_mode", type=str, default="end-to-end", help="Finetuning mode"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        help="Name of dataset(s)",
        default="/TimeseriesDatasets/forecasting/monash/m3_monthly_dataset.tsf",
    )

    args = parser.parse_args()

    forecasting(
        config_path=args.config,
        gpu_id=args.gpu_id,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        finetuning_mode=args.finetuning_mode,
        init_lr=args.init_lr,
        max_epoch=args.max_epoch,
        dataset_names=args.dataset_names,
    )
