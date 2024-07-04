import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from moment.utils.anomaly_detection_metrics import get_anomaly_detection_metrics
from moment.utils.utils import dtype_map, make_dir_if_not_exists

from .base import Tasks

warnings.filterwarnings("ignore")


class AnomlayDetectionFinetuning(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self.criterion = self._select_criterion(
            loss_type=self.args.loss_type, reduction="mean"
        )

    def validation(self, data_loader, return_preds: bool = False):
        trues, preds, losses = [], [], []

        self.model.eval()
        with torch.set_grad_enabled(self.args.enable_val_grad):
            for batch_x in tqdm(data_loader, total=len(data_loader)):
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model.detect_anomalies(
                        x_enc=timeseries,
                        input_mask=input_mask,
                        mask=None,
                        anomaly_criterion=self.args.anomaly_criterion,
                    )

                loss = self.criterion(outputs.reconstruction, timeseries)
                losses.append(loss.item())

                if return_preds:
                    trues.append(timeseries.detach().cpu().numpy())
                    preds.append(outputs.reconstruction.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)

        self.model.train()

        if return_preds:
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)

            return average_loss, losses, (trues, preds)
        else:
            return average_loss

    def train(self):
        # Setup logger
        self.logger = self.setup_logger()
        self.run_name = self.logger.name

        # Make necessary directories for logging and saving
        self.checkpoint_path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(self.checkpoint_path, verbose=True)
        self.optimizer = self._select_optimizer()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler(type=self.args.lr_scheduler_type)

        # Load pre-trained MOMENT model before fine-tuning
        if self.args.model_name == "MOMENT":
            self.load_pretrained_moment(
                do_not_copy_head=False
            )  # For reconstruction based tasks we can load the head too

        self.model.to(self.device)

        # During training, take smaller strides
        self.args.data_stride_len = 1

        self.results_dir = self._create_results_dir(
            experiment_name="supervised_anomaly_detection"
        )

        # Evaluate the models before training
        eval_metrics = self.evaluate_model()
        self.logger.log(
            {
                "train_loss": eval_metrics.train_loss,
                "validation_loss": eval_metrics.val_loss,
                "test_loss": eval_metrics.test_loss,
            }
        )

        opt_steps = 0
        cur_epoch = 0
        best_validation_loss = np.inf

        while cur_epoch < self.args.max_epoch:  # Epoch based learning only
            self.model.train()

            for batch_x in tqdm(
                self.train_dataloader, total=len(self.train_dataloader)
            ):
                self.optimizer.zero_grad(set_to_none=True)
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries,
                        input_mask=input_mask,
                        mask=None,
                        anomaly_criterion=self.args.anomaly_criterion,
                    )

                loss = self.criterion(outputs.reconstruction, timeseries)

                self.logger.log(
                    {
                        "step_train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

                if self.args.debug and opt_steps >= 1:
                    self.debug_model_outputs(loss, outputs, batch_x)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.scaler.step(self.optimizer)

                # Updates the scale for next iteration.
                self.scaler.update()

                opt_steps = opt_steps + 1

                # Adjust learning rate
                if self.args.lr_scheduler_type == "linearwarmupcosinelr":
                    self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
                elif (
                    self.args.lr_scheduler_type == "onecyclelr"
                ):  # Should be torch schedulers in general
                    self.lr_scheduler.step()

            cur_epoch = cur_epoch + 1

            eval_metrics = self.evaluate_and_log()

            if eval_metrics.val_loss < best_validation_loss:
                best_validation_loss = eval_metrics.val_loss
                self.save_model_and_alert(opt_steps=None)

        return self.model

    def test(self):
        # Load model
        checkpoint = torch.load(
            os.path.join(self.checkpoint_path, f"{self.args.model_name}.pth"),
            map_location=lambda storage, loc: storage.cuda(self.device),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # During testing, each time-step in a window is seen only once
        self._reset_dataloader()
        self.args.data_stride_len = self.args.seq_len
        self.args.shuffle = False  # We must not shuffle during testing
        self.test_dataloader = self._get_dataloader(data_split="test")

        # Get test dataloader and evaluate the model
        trues, preds, labels = [], [], []

        self.model.eval()
        with torch.set_grad_enabled(self.args.enable_val_grad):
            for batch_x in tqdm(self.test_dataloader, total=len(self.test_dataloader)):
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)
                labels.append(batch_x.labels)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model.detect_anomalies(
                        x_enc=timeseries,
                        input_mask=input_mask,
                        mask=None,
                        anomaly_criterion=self.args.anomaly_criterion,
                    )

                trues.append(timeseries.detach().cpu().numpy())
                preds.append(outputs.reconstruction.detach().cpu().numpy())

        # NOTE: Assuming anomaly detection datasets only have 1 channel/feature
        trues = np.concatenate(trues, axis=0).squeeze().flatten()
        preds = np.concatenate(preds, axis=0).squeeze().flatten()
        labels = np.concatenate(labels, axis=0).squeeze().flatten()

        # Get anomaly detection metrics
        anomaly_scores = (trues - preds) ** 2
        len_timeseries = self.test_dataloader.dataset.length_timeseries
        print(
            f"Anomaly scores: {anomaly_scores.shape} | Labels: {labels.shape} | Timeseries: {len_timeseries}"
        )

        metrics = get_anomaly_detection_metrics(
            anomaly_scores=anomaly_scores, labels=labels
        )

        results_df = pd.DataFrame(
            data=[
                self.run_name,
                self.logger.id,
                metrics.adjbestf1,
                metrics.raucroc,
                metrics.raucpr,
                metrics.vusroc,
                metrics.vuspr,
            ],
            index=[
                "Model name",
                "ID",
                "Adj. Best F1",
                "rAUCROC",
                "rAUCPR",
                "VUSROC",
                "VUSPR",
            ],
        )

        metadata = self.args.dataset_names.split("/")[-1].split("_")
        data_id, data_name = metadata[0], metadata[3]

        results_df.to_csv(
            os.path.join(self.results_dir, f"results_{data_id}_{data_name}.csv")
        )

        return True
