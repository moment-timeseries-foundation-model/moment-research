import argparse
import os

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from moment.common import PATHS
from moment.data.dataloader import get_timeseries_dataloader
from moment.models.base import BaseModel
from moment.models.moment import MOMENT
from moment.utils.anomaly_detection_metrics import get_anomaly_detection_metrics
from moment.utils.config import Config
from moment.utils.ucr_anomaly_archive_subset import ucr_anomaly_archive_subset
from moment.utils.utils import control_randomness, dtype_map, parse_config


def test(args, model, results_dir) -> pd.DataFrame:
    # Get test dataloader
    test_dataloader = get_test_dataloader(args)

    # Evaluate the model
    anomaly_scores, trues, preds, labels = [], [], [], []

    model.eval()
    with torch.set_grad_enabled(args.enable_val_grad):
        for batch_x in tqdm(
            test_dataloader,
            total=len(test_dataloader),
            disable=(not args.enable_batchwise_pbar),
        ):
            timeseries = batch_x.timeseries.float().to(args.device)
            input_mask = batch_x.input_mask.long().to(args.device)
            labels.append(batch_x.labels)

            with torch.autocast(
                device_type="cuda",
                dtype=dtype_map(args.torch_dtype),
                enabled=args.use_amp,
            ):
                outputs = model.detect_anomalies(
                    x_enc=timeseries,
                    input_mask=input_mask,
                    anomaly_criterion=args.anomaly_criterion,
                )

            anomaly_scores.append(outputs.anomaly_scores.detach().cpu().numpy())
            preds.append(outputs.reconstruction.detach().cpu().numpy())
            trues.append(timeseries.detach().cpu().numpy())

        # NOTE: Assuming anomaly detection datasets only have 1 channel/feature
        anomaly_scores = np.concatenate(anomaly_scores, axis=0).squeeze().flatten()
        preds = np.concatenate(preds, axis=0).squeeze().flatten()
        trues = np.concatenate(trues, axis=0).squeeze().flatten()
        labels = np.concatenate(labels, axis=0).squeeze().flatten()

    if args.anomaly_criterion == "mse":
        np.allclose(anomaly_scores, ((trues - preds) ** 2))  # Sanity check

    len_timeseries = test_dataloader.dataset.length_timeseries
    # print(f"Anomaly scores: {anomaly_scores.shape} | Labels: {labels.shape} | Timeseries: {len_timeseries}")

    metrics = get_anomaly_detection_metrics(
        anomaly_scores=anomaly_scores, labels=labels, n_splits=args.n_splits
    )

    results_df = pd.DataFrame(
        data=[
            metrics.adjbestf1,
            metrics.raucroc,
            metrics.raucpr,
            metrics.vusroc,
            metrics.vuspr,
        ],
        index=["Adj. Best F1", "rAUCROC", "rAUCPR", "VUSROC", "VUSPR"],
    )

    metadata = args.dataset_names.split("/")[-1].split("_")
    data_id, data_name = metadata[0], metadata[3]

    results_df.to_csv(os.path.join(results_dir, f"results_{data_id}_{data_name}.csv"))
    return results_df


def get_test_dataloader(args):
    args.dataset_names = args.full_file_path_and_name
    args.data_split = "test"
    args.batch_size = args.val_batch_size
    args.data_stride_len = (
        args.seq_len
    )  # Such that each timestep in a window is seen only once
    args.shuffle = False  # We must not shuffle during testing
    test_dataloader = get_timeseries_dataloader(args=args)
    return test_dataloader


def load_pretrained_model(args):
    checkpoint = BaseModel.load_pretrained_weights(
        run_name=args.pretraining_run_name, opt_steps=args.pretraining_opt_steps
    )
    model = MOMENT(configs=args)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _create_results_dir(args, experiment_name="zero_shot_anomaly_detection"):
    results_path = os.path.join(PATHS.RESULTS_DIR, experiment_name)

    os.makedirs(results_path, exist_ok=True)
    return results_path


def run_experiment(
    config_path: str = "configs/anomaly_detection/zero_shot.yaml",
    default_config_path: str = "configs/default.yaml",
    experiment_name: str = "zero_shot_anomaly_detection",
    pretraining_run_name: str = None,
    opt_steps: int = None,
    run_name: str = None,
    gpu_id: int = 0,
):
    # Load arguments and parse them
    config = Config(
        config_file_path=config_path, default_config_file_path=default_config_path
    ).parse()
    args = parse_config(config)

    # Control randomness
    control_randomness(config["random_seed"])

    # Set-up parameters and defaults
    config["device"] = (
        torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
    )
    args = parse_config(config)

    # Save the experiment arguments and metadata
    results_dir = _create_results_dir(args=args, experiment_name=experiment_name)

    # Load model
    if pretraining_run_name:
        print(f"Setting pretraining_run_name to {pretraining_run_name}")
        args.pretraining_run_name = pretraining_run_name
    if opt_steps:
        print(f"Setting pretraining_opt_steps to {opt_steps}")
        args.pretraining_opt_steps = opt_steps
    model = load_pretrained_model(args)
    model.to(args.device)

    all_anomaly_detection_datasets = ucr_anomaly_archive_subset

    datasets_with_failed_experiments = []

    print(f"Total datasets: {len(all_anomaly_detection_datasets)}")
    pbar = tqdm(
        all_anomaly_detection_datasets[:4], total=len(all_anomaly_detection_datasets)
    )

    results = {}
    for full_file_path_and_name in pbar:
        dataset_name = full_file_path_and_name.split("/")[-1][:-4]
        pbar.set_postfix({"Dataset": dataset_name})
        args.full_file_path_and_name = full_file_path_and_name

        result_df = test(model=model, args=args, results_dir=results_dir)
        results[dataset_name] = result_df.T

    # log the results
    anomaly_metrics_all = pd.concat(results.values(), keys=results.keys(), axis=0)
    anomaly_metrics_all = (
        anomaly_metrics_all.reset_index(level=0)
        .rename(columns={"level_0": "Dataset"})
        .dropna()
    )

    anomaly_metrics_mean = anomaly_metrics_all.drop(["Dataset"], axis=1)
    anomaly_metrics_mean = anomaly_metrics_mean.mean(axis=0).to_frame()

    logger = wandb.init(
        project="Time-series Foundation Model",
        dir=PATHS.WANDB_DIR,
        config=args,
        name=run_name,
        mode="disabled" if args.debug else "run",
    )
    logger.log(
        {
            "anomaly_metrics_all": wandb.Table(
                columns=anomaly_metrics_all.columns.tolist(),
                data=anomaly_metrics_all.values.tolist(),
            ),
            "anomaly_metrics_mean": wandb.Table(
                columns=anomaly_metrics_mean.T.columns.tolist(),
                data=anomaly_metrics_mean.T.values.tolist(),
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, default="zero_shot_anomaly_detection"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/anomaly_detection/zero_shot_anomaly_detection.yaml",
    )
    parser.add_argument("--pretraining_run_name", type=str, default=None)
    parser.add_argument("--opt_steps", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()
    run_experiment(
        config_path=args.config_path,
        experiment_name=args.experiment_name,
        pretraining_run_name=args.pretraining_run_name,
        opt_steps=args.opt_steps,
        run_name=args.run_name,
        gpu_id=args.gpu_id,
    )
