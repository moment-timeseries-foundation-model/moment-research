import argparse
import datetime
import os
import pickle as pkl

import numpy as np
import torch
from tqdm import tqdm
from yaml import dump

from moment.common import PATHS
from moment.data.base import ClassificationResults
from moment.data.dataloader import get_timeseries_dataloader
from moment.models.base import BaseModel
from moment.models.moment import MOMENT
from moment.models.statistical_classifiers import fit_svm
from moment.utils.config import Config
from moment.utils.uea_classification_datasets import uea_classification_datasets
from moment.utils.utils import control_randomness, parse_config

DEFAULT_CONFIG_PATH = "configs/default.yaml"

SMALL_IMAGE_DATASETS = [
    "Crop",
    "MedicalImages",
    "SwedishLeaf",
    "FacesUCR",
    "FaceAll",
    "Adiac",
    "ArrowHead",
]
SMALL_SPECTRO_DATASETS = ["Wine", "Strawberry", "Coffee", "Ham", "Meat", "Beef"]


def get_embeddings_and_labels(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    enable_batchwise_pbar: bool = False,
):
    model = model.to(device)
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for batch_x in tqdm(
            dataloader, total=len(dataloader), disable=(not enable_batchwise_pbar)
        ):
            timeseries = batch_x.timeseries.float().to(device)
            input_mask = batch_x.input_mask.long().to(device)

            outputs = model.embed(
                x_enc=timeseries, input_mask=input_mask, reduction="mean"
            )

            embeddings_ = outputs.embeddings.detach().cpu().numpy()
            embeddings.append(embeddings_)
            labels.append(batch_x.labels)

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0).squeeze()

    return embeddings, labels


def get_dataloaders(args):
    args.dataset_names = args.full_file_path_and_name
    args.data_split = "train"
    train_dataloader = get_timeseries_dataloader(args=args)
    args.data_split = "test"
    test_dataloader = get_timeseries_dataloader(args=args)
    args.data_split = "val"
    val_dataloader = get_timeseries_dataloader(args=args)
    return train_dataloader, test_dataloader, val_dataloader


def load_pretrained_moment(args, pretraining_task_name: str = "pre-training"):
    args.task_name = pretraining_task_name

    checkpoint = BaseModel.load_pretrained_weights(
        run_name=args.pretraining_run_name, opt_steps=args.pretraining_opt_steps
    )

    pretrained_model = MOMENT(configs=args)
    pretrained_model.load_state_dict(checkpoint["model_state_dict"])

    return pretrained_model


def _create_results_dir(experiment_name):
    results_path = os.path.join(PATHS.RESULTS_DIR, experiment_name)
    os.makedirs(results_path, exist_ok=True)
    return results_path


def _save_config(args, results_path):
    with open(os.path.join(results_path, "config.yaml"), "w") as f:
        dump(vars(args), f)


def run_experiment(
    experiment_name: str = "unsupervised_representation_learning",
    config_path: str = None,
    gpu_id: str = "0",
):
    # Load arguments and parse them
    config = Config(
        config_file_path=config_path, default_config_file_path=DEFAULT_CONFIG_PATH
    ).parse()

    config["device"] = (
        torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
    )
    args = parse_config(config)

    # Set-up parameters and defaults
    args.config_file_path = config_path
    args.shuffle = False
    args.run_datetime = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")

    # Set all randomness
    control_randomness(seed=args.random_seed)

    # Save the experiment arguments and metadata
    results_path = _create_results_dir(experiment_name)
    _save_config(args, results_path)

    model = load_pretrained_moment(args)
    all_classification_datasets = uea_classification_datasets

    datasets_with_failed_experiments = []

    pbar = tqdm(all_classification_datasets, total=len(all_classification_datasets))
    for full_file_path_and_name in pbar:
        dataset_name = full_file_path_and_name.split("/")[-2]
        pbar.set_postfix({"Dataset": dataset_name})

        if (dataset_name in SMALL_IMAGE_DATASETS) or (
            dataset_name in SMALL_SPECTRO_DATASETS
        ):
            args.upsampling_type = "interpolate"
        else:
            args.upsampling_type = "pad"

        args.full_file_path_and_name = full_file_path_and_name
        args.task_name = "classification"

        train_dataloader, test_dataloader, val_dataloader = get_dataloaders(args)

        train_embeddings, train_labels = get_embeddings_and_labels(
            model=model,
            dataloader=train_dataloader,
            device=torch.device(args.device),
            enable_batchwise_pbar=False,
        )

        test_embeddings, test_labels = get_embeddings_and_labels(
            model=model,
            dataloader=test_dataloader,
            device=torch.device(args.device),
            enable_batchwise_pbar=False,
        )

        val_embeddings, val_labels = get_embeddings_and_labels(
            model=model,
            dataloader=val_dataloader,
            device=torch.device(args.device),
            enable_batchwise_pbar=False,
        )

        train_embeddings = np.concatenate([train_embeddings, val_embeddings], axis=0)
        train_labels = np.concatenate([train_labels, val_labels], axis=0)

        # Reshape embeddings and labels
        train_embeddings = train_embeddings.reshape(
            (-1, train_dataloader.dataset.n_channels, args.d_model)
        )
        train_labels = train_labels.reshape((len(train_embeddings), -1)).astype(int)
        test_embeddings = test_embeddings.reshape(
            (-1, test_dataloader.dataset.n_channels, args.d_model)
        )
        test_labels = test_labels.reshape((len(test_embeddings), -1)).astype(int)

        # Average across channels
        # train_embeddings = train_embeddings.mean(axis=1)
        train_embeddings = torch.flatten(
            torch.tensor(train_embeddings), start_dim=1
        ).numpy()
        train_labels = train_labels.mean(axis=1)
        # test_embeddings = test_embeddings.mean(axis=1)
        test_embeddings = torch.flatten(
            torch.tensor(test_embeddings), start_dim=1
        ).numpy()
        test_labels = test_labels.mean(axis=1)

        # Fit SVM
        try:
            classifier = fit_svm(features=train_embeddings, y=train_labels)
        except:
            datasets_with_failed_experiments.append(dataset_name)
            continue

        # Evaluate the model
        y_pred_train = classifier.predict(train_embeddings)
        y_pred_test = classifier.predict(test_embeddings)
        train_accuracy = classifier.score(train_embeddings, train_labels)
        test_accuracy = classifier.score(test_embeddings, test_labels)

        result_object = ClassificationResults(
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            train_labels=train_labels,
            test_labels=test_labels,
            train_predictions=y_pred_train,
            test_predictions=y_pred_test,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            dataset_name=dataset_name.split("/")[-1],
        )

        with open(os.path.join(results_path, f"results_{dataset_name}.pkl"), "wb") as f:
            pkl.dump(result_object, f)

        with open(
            os.path.join(results_path, "datasets_with_failed_experiments.txt"), "w"
        ) as f:
            f.write("\n".join(datasets_with_failed_experiments))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, default="unsupervised_representation_learning"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/classification/unsupervised_representation_learning.yaml",
    )
    parser.add_argument("--gpu_id", type=str, default="0")

    args = parser.parse_args()

    run_experiment(
        experiment_name=args.experiment_name,
        config_path=args.config_path,
        gpu_id=args.gpu_id,
    )
