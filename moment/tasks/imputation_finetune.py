import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from moment.utils.forecasting_metrics import get_forecasting_metrics
from moment.utils.utils import dtype_map, make_dir_if_not_exists

from .base import Tasks

warnings.filterwarnings("ignore")


class ImputationFinetuning(Tasks):
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
                    outputs = self.model.pretraining(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                recon_loss = self.criterion(outputs.reconstruction, timeseries)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                n_channels = outputs.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

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
        self.dataset_name_ = self.args.dataset_names.split("/")[-1].split(".")[0]

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
            experiment_name="supervised_imputation"
        )

        # Evaluate the models before training
        eval_metrics = self.evaluate_model()
        self.logger.log(
            {
                # "train_loss": eval_metrics.train_loss,
                "validation_loss": eval_metrics.val_loss,
                # "test_loss": eval_metrics.test_loss
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
                    # Masked time-series prediction
                    outputs = self.model.pretraining(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                recon_loss = self.criterion(outputs.reconstruction, timeseries)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                n_channels = outputs.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

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

    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        self.logger.log(
            {
                # "train_loss": eval_metrics.train_loss,
                "validation_loss": eval_metrics.val_loss,
                # "test_loss": eval_metrics.test_loss
            }
        )
        return eval_metrics
