import argparse
import sys
import time
from typing import Callable

import numpy as np
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from moment.data.bit_datasets import BitMemoryDataset, BitXORDataset
from moment.data.image_datasets import CIFAR10Dataset, CIFAR10GrayDataset, MNISTDataset
from moment.data.nlp_datasets import NLPDataset
from moment.models.fpt import FrozenPretrainedTransformer
from moment.utils.config import Config
from moment.utils.utils import control_randomness, parse_config
from scripts.development.fpt_trainer import Trainer


def get_score_functions(dataset_name: str) -> tuple[Callable, Callable]:
    ce_loss = torch.nn.CrossEntropyLoss()

    if dataset_name == "CIFAR10" or dataset_name == "MNIST":
        # original - CIFAR10, MNIST
        def loss_fn(out, y, x=None):
            out = out[:, 0]
            return ce_loss(out, y)

        def accuracy_fn(preds, true, x=None):
            preds = preds[:, 0].argmax(-1)
            return (preds == true).mean()

        return loss_fn, accuracy_fn

    elif dataset_name == "IMDB":
        # IMDB
        def loss_fn(out, y, x=None):
            return ce_loss(out, y)

        def accuracy_fn(preds, true, x=None):
            pred_labels = np.argmax(preds, axis=1)
            correct = np.sum(pred_labels == true)
            accuracy = correct / true.shape[0]
            return accuracy

        return loss_fn, accuracy_fn

    elif dataset_name == "BitMemory":
        # Bit-Memory
        def loss_fn(out, y, x=None):
            out = torch.reshape(out, (-1, 1000, 2))
            ids = torch.zeros(y.shape).to(device=y.device).long()
            ids[y < 0], ids[y > 0] = 0, 1
            out, ids = torch.reshape(out, (-1, 2)), torch.reshape(ids, (-1,))
            return ce_loss(out, ids)

        def accuracy_fn(preds, true, x=None):
            preds = preds.reshape(-1, 1000, 2).argmax(-1) * 2 - 1
            return (np.sign(preds) == np.sign(true)).mean()

        return loss_fn, accuracy_fn

    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")


def train(
    config_file_path: str,
    dataset_name: str,
    input_dim: int,
    output_dim: int,
    use_embeddings_for_input: bool,
    GPU_ID: int,
):
    print(f"config: {config_file_path}\ndataset: {dataset_name}\ngpu: {GPU_ID}\n")

    # Load config
    DEFAULT_CONFIG_PATH = "configs/default.yaml"
    config = Config(
        config_file_path=config_file_path, default_config_file_path=DEFAULT_CONFIG_PATH
    ).parse()
    config["device"] = GPU_ID if torch.cuda.is_available() else "cpu"
    args = parse_config(config)
    control_randomness(args.random_seed)

    # Update args
    args.dataset_name = dataset_name
    args.input_dim = input_dim
    args.output_dim = output_dim
    args.use_embeddings_for_input = use_embeddings_for_input

    # Device
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    # Load model
    FPT = FrozenPretrainedTransformer(configs=args)
    FPT = FPT.to(device)
    if sys.version_info <= (3, 10):
        print("Compiling FPT model...")
        FPT = torch.compile(FPT)

    # Load dataset
    if dataset_name == "CIFAR10":
        dataset = CIFAR10Dataset(
            batch_size=args.batch_size, patch_size=4, device=device
        )
    elif dataset_name == "MNIST":
        dataset = MNISTDataset(batch_size=args.batch_size, patch_size=4, device=device)
    elif dataset_name == "BitMemory":
        dataset = BitMemoryDataset(n=1000, num_patterns=5, device=device)
    elif dataset_name == "IMDB":
        dataset = NLPDataset(
            dataset_name="imdb",
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")

    x, y = dataset.get_batch(batch_size=args.batch_size)
    print(f"x shape: {x.shape} y shape: {y.shape}")

    # Loss function
    loss_fn, accuracy_fn = get_score_functions(dataset_name)

    # Train
    trainer = Trainer(
        FPT,
        dataset,
        loss_fn,
        accuracy_fn=accuracy_fn,
        steps_per_epoch=args.steps_per_epoch,
        test_steps_per_epoch=int(args.steps_per_epoch * 0.2),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=1,
        grad_accumulate=1,
    )

    wandb.init(
        project="Time-series Foundation Model",
        name=args.run_name,
        config=args,
        mode="disabled" if args.debug else "run",
    )

    total_steps = 0
    for i in range(args.num_epochs):
        trainer.train_epoch()
        total_steps += args.steps_per_epoch
        wandb.log(
            {
                "Train Loss": trainer.diagnostics["Average Train Loss"],
                "Test Loss": trainer.diagnostics["Test Loss"],
                "Train Accuracy": trainer.diagnostics["Train Accuracy"],
                "Test Accuracy": trainer.diagnostics["Test Accuracy"],
                "Epoch": i,
                "Steps": total_steps,
            }
        )
        print(
            f'Epoch {i+1}/{args.num_epochs} '
            f'Train Loss: {trainer.diagnostics["Average Train Loss"]:.3f} '
            f'Test Loss: {trainer.diagnostics["Test Loss"]:.3f} '
            f'Train Accuracy: {trainer.diagnostics["Train Accuracy"]:.3f} '
            f'Test Accuracy: {trainer.diagnostics["Test Accuracy"]:.3f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="../../configs/default.yaml", help="config"
    )
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset name")
    parser.add_argument("--input_dim", type=int, default=48, help="input dimension")
    parser.add_argument("--output_dim", type=int, default=10, help="output dimension")
    parser.add_argument(
        "--use_embeddings_for_input",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="only True for IMDB",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    experiment_args = parser.parse_args()
    train(
        experiment_args.config,
        experiment_args.dataset,
        experiment_args.input_dim,
        experiment_args.output_dim,
        experiment_args.use_embeddings_for_input,
        experiment_args.gpu_id,
    )
