import os
import subprocess
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from wandb import AlertLevel

from moment.common import PATHS
from moment.models.moment import MOMENT
from moment.utils.utils import MetricsStore, dtype_map, make_dir_if_not_exists

from .base import Tasks

warnings.filterwarnings("ignore")


class Pretraining(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args

    def validation(self, data_loader, return_preds: bool = False):
        trues, preds, masks, losses = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader)):
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
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
                    masks.append(outputs.pretrain_mask.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        self.model.train()

        if return_preds:
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            masks = np.concatenate(masks, axis=0)
            return average_loss, losses, (trues, preds, masks)
        else:
            return average_loss

    def train(self):
        self.run_name = self.logger.name
        path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(path, verbose=True)

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler()

        self.model.to(self.device)
        # self.evaluate_and_log()

        opt_steps = 0
        cur_epoch = 0
        while opt_steps < self.args.max_opt_steps or cur_epoch < self.args.max_epoch:
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
                self.scaler.update()
                opt_steps = opt_steps + 1

                if opt_steps % self.args.log_interval == 0:
                    self.evaluate_and_log()

                if opt_steps % self.args.checkpoint_interval == 0:
                    self.logger.alert(
                        title="Saving model",
                        text=f"Saving model after {opt_steps} steps",
                        level=AlertLevel.INFO,
                    )
                    self.save_model(
                        self.model, path, opt_steps, self.optimizer, self.scaler
                    )
                    self.evaluate_model_external(path, opt_steps, self.device)

                self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)

            cur_epoch = cur_epoch + 1

        return self.model

    def evaluate_model(self):
        return MetricsStore(val_loss=self.validation(self.val_dataloader))

    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        self.logger.log({"validation_loss": eval_metrics.val_loss})
        return eval_metrics

    def evaluate_model_external(self, path: str, opt_steps: int, device) -> None:
        print("starting evaluation")
        eval_device = int(str(device).split(":")[-1]) + 1
        command = [
            "python",
            "scripts/evaluation/evaluation.py",
            f"--checkpoint_path={path}",
            f"--opt_steps={opt_steps}",
            f"--run_name={self.run_name}",
            f"--gpu_id={eval_device}",
        ]
        outfile = open(f"{path}/eval_output.txt", "w")
        errfile = open(f"{path}/eval_error.txt", "w")
        subprocess.Popen(command, stdout=outfile, stderr=errfile)
