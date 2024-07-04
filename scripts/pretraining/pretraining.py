import argparse

import torch

from moment.common import PATHS
from moment.tasks.pretrain import Pretraining
from moment.utils.config import Config
from moment.utils.utils import control_randomness, make_dir_if_not_exists, parse_config


def pretrain(
    config_path: str = "configs/pretraining/pretrain.yaml",
    default_config_path: str = "configs/default.yaml",
    gpu_id: int = 0,
) -> None:
    config = Config(
        config_file_path=config_path, default_config_file_path=default_config_path
    ).parse()

    control_randomness(config["random_seed"])

    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    args = parse_config(config)
    make_dir_if_not_exists(config["checkpoint_path"])

    print(f"Running experiments with config:\n{args}\n")
    task_obj = Pretraining(args=args)

    NOTES = "Pre-training runs"
    task_obj.setup_logger(notes=NOTES)
    task_obj.train()
    task_obj.end_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretraining/pretrain.yaml",
        help="Path to config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    pretrain(config_path=args.config, gpu_id=args.gpu_id)
