import os
from typing import Optional

import torch
import torch.nn as nn

from moment.common import PATHS, TASKS


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_enc: torch.Tensor, input_mask: torch.Tensor = None):
        """
        Forward pass of the model.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input tensor of shape (batch_size, n_channels, seq_len)
        input_mask : torch.Tensor, optional
            Input mask of shape (batch_size, seq_len), by default None

        Returns
        -------
        TimeseriesOutputs
        """
        if input_mask is None:
            batch_size, _, seq_len = x_enc.shape
            input_mask = torch.ones((batch_size, seq_len))

        if (
            self.task_name == TASKS.LONG_HORIZON_FORECASTING
            or self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        ):
            return self.forecast(x_enc, input_mask)
        elif self.task_name == TASKS.IMPUTATION:
            return self.imputation(x_enc, input_mask)
        elif self.task_name == TASKS.ANOMALY_DETECTION:
            dec_out = self.anomaly_detection(x_enc, input_mask)
            return dec_out
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classification(x_enc, input_mask)
        elif self.task_name == TASKS.PRETRAINING:
            return self.pretraining(x_enc, input_mask)
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc, input_mask)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")

    @staticmethod
    def load_pretrained_weights(
        run_name: str,
        opt_steps: Optional[int] = None,
        checkpoints_dir: str = PATHS.CHECKPOINTS_DIR,
    ):
        path = os.path.join(checkpoints_dir, run_name)

        if opt_steps is None:
            opt_steps = [int(i.split("_")[-1].split(".")[0]) for i in os.listdir(path)]
            opt_steps = max(opt_steps)
            print(f"Loading latest model checkpoint at {opt_steps} steps")

        checkpoint_path = os.path.join(path, f"moment_checkpoint_{opt_steps}.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(path, f"MOMENT_checkpoint_{opt_steps}.pth")

        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        return checkpoint
