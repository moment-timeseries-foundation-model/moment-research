import argparse
from typing import Optional

import torch

from moment.common import PATHS
from moment.tasks.imputation_finetune import ImputationFinetuning
from moment.utils.config import Config
from moment.utils.utils import control_randomness, make_dir_if_not_exists, parse_config

NOTES = "Pre-training GPT4TS for imputation"


def imputation(
    config_path: str = "../../configs/imputation/gpt4ts_train.yaml",
    default_config_path: str = "../../configs/default.yaml",
    gpu_id: int = 0,
    train_batch_size: int = 64,
    val_batch_size: int = 256,
    init_lr: Optional[float] = None,
    n_channels: int = 7,
    dataset_names: str = "/TimeseriesDatasets/forecasting/autoformer/electricity.csv",
) -> None:
    config = Config(
        config_file_path=config_path, default_config_file_path=default_config_path
    ).parse()

    # Control randomness
    control_randomness(config["random_seed"])

    # Set-up parameters and defaults
    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    args = parse_config(config)
    make_dir_if_not_exists(config["checkpoint_path"])

    # Setup arguments
    args.train_batch_size = train_batch_size
    args.val_batch_size = val_batch_size
    args.finetuning_mode = "end-to-end"
    args.dataset_names = dataset_names
    args.n_channels = n_channels
    if init_lr is not None:
        args.init_lr = init_lr

    print(f"Running experiments with config:\n{args}\n")

    task_obj = ImputationFinetuning(args=args)

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
    parser.add_argument("--n_channels", type=int, default=7, help="Number of channels")
    parser.add_argument(
        "--dataset_names",
        type=str,
        help="Name of dataset(s)",
        default="/TimeseriesDatasets/forecasting/autoformer/electricity.csv",
    )

    args = parser.parse_args()

    imputation(
        config_path=args.config,
        gpu_id=args.gpu_id,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        init_lr=args.init_lr,
        n_channels=args.n_channels,
        dataset_names=args.dataset_names,
    )
