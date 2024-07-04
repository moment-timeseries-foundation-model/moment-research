"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moment.common import TASKS
from moment.data.base import TimeseriesOutputs
from moment.models.base import BaseModel
from moment.utils.utils import get_anomaly_criterion


@dataclass
class DGHLOutputs:
    timeseries: torch.Tensor = None
    reconstruction: torch.Tensor = None
    mask: torch.Tensor = None
    loss: torch.Tensor = None


class DGHL(BaseModel):
    """
    DGHL: Deep Generative model with Hierarchical Latent Factors for Time Series Anomaly Detection

    References
    -----------
    [1] Challu, Cristian I., et al. "Deep generative model with hierarchical
        latent factors for time series anomaly detection." International
        Conference on Artificial Intelligence and Statistics. PMLR, 2022.
        https://arxiv.org/abs/2202.07586
    """

    def __init__(self, configs):
        super(DGHL, self).__init__()

        self._validate_inputs(configs)

        self.configs = configs

        self.window_size = configs.seq_len
        self.window_step = self.window_size
        self.device = configs.device
        self.task_name = configs.task_name

        # Generator
        self.n_features = configs.n_channels
        self.hidden_multiplier = configs.hidden_multiplier
        self.z_size = configs.z_size
        self.z_size_up = configs.z_size_up
        self.max_filters = configs.max_filters
        self.kernel_multiplier = configs.kernel_multiplier
        self.normalize_windows = configs.normalize_windows
        self.a_L = configs.a_L

        # Alternating back-propagation
        self.z_iters = configs.z_iters
        self.z_iters_inference = configs.z_iters_inference
        self.z_sigma = configs.z_sigma
        self.z_step_size = configs.z_step_size
        self.z_with_noise = configs.z_with_noise
        self.z_persistent = configs.z_persistent
        self.p_0_chains_u = None
        self.p_0_chains_l = None

        # Training
        self.noise_std = configs.noise_std

        # Generator
        # torch.manual_seed(random_seed)

        self.sub_window_size = int(self.window_size / configs.a_L)

        # Learnable parameters should be in self.model
        self.model = Generator(
            window_size=self.sub_window_size,
            hidden_multiplier=self.hidden_multiplier,
            latent_size=self.z_size + self.z_size_up,
            n_features=self.n_features,
            max_filters=self.max_filters,
            kernel_multiplier=self.kernel_multiplier,
        ).to(self.device)

        self.mse_loss = nn.MSELoss(reduction="sum")

    def _validate_inputs(self, configs):
        assert configs.z_persistent == False, "z-persistence not implemented yet"

    def infer_z(self, z, Y, mask, n_iters, with_noise):
        z_u = z[0]
        z_l = z[1]

        for i in range(n_iters):
            z_u = torch.autograd.Variable(z_u, requires_grad=True)
            z_l = torch.autograd.Variable(z_l, requires_grad=True)

            z_u_repeated = torch.repeat_interleave(z_u, self.a_L, 0)
            z = torch.cat((z_u_repeated, z_l), dim=1).to(self.device)

            Y_hat = self.model(z, mask)

            L = 1.0 / (2.0 * self.z_sigma * self.z_sigma) * self.mse_loss(Y_hat, Y)
            L.backward()
            z_u = z_u - 0.5 * self.z_step_size * self.z_step_size * (z_u + z_u.grad)
            z_l = z_l - 0.5 * self.z_step_size * self.z_step_size * (z_l + z_l.grad)
            if with_noise:
                eps_u = torch.randn(len(z_u), self.z_size_up, 1, 1).to(z_u.device)
                z_u += self.z_step_size * eps_u
                eps_l = torch.randn(len(z_l), self.z_size, 1, 1).to(z_l.device)
                z_l += self.z_step_size * eps_l

        z_u = z_u.detach()
        z_l = z_l.detach()
        z = z.detach()

        return z, z_u, z_l

    def sample_gaussian(self, n_dim, n_samples):
        p_0 = torch.distributions.MultivariateNormal(
            torch.zeros(n_dim), 0.01 * torch.eye(n_dim)
        )
        p_0 = p_0.sample([n_samples]).view([n_samples, -1, 1])

        return p_0

    def _preprocess_batch_hierarchy(self, windows):
        """
        X tensor of shape (batch_size, n_features, window_size*a_L)
        """
        batch_size, n_features, window_size = windows.shape

        assert (
            n_features == self.n_features
        ), f"Batch n_features {n_features} not consistent with Generator"
        assert (
            window_size == self.window_size
        ), f"Window size {window_size} not consistent with Generator"

        # Wrangling from (batch_size, n_features, window_size*window_hierarchy) -> (batch_size*window_hierarchy, n_features, window_size)
        # (batch_size, n_features, 1, window_size*a_L)
        windows = windows.unfold(
            dimension=-1, size=self.sub_window_size, step=self.sub_window_size
        )

        # (batch_size, n_features, a_L, sub_window_size), such that a_L*sub_window_size = window_size
        windows = windows.swapaxes(1, 2)
        # (batch_size, a_L, n_features, sub_window_size)

        windows = windows.reshape(
            batch_size * self.a_L, self.n_features, self.sub_window_size
        )
        # (batch_size*a_L, n_features, sub_window_size)

        return windows

    def _postprocess_batch_hierarchy(self, windows):
        # (batch_size*a_L, n_features, sub_window_size)

        # Return to window_size * window_hierarchy size
        windows = windows.swapaxes(0, 1)
        # (n_features, batch_size*a_L, sub_window_size)
        windows = windows.reshape(self.n_features, -1, self.window_size)
        # (n_features, batch_size, window_size)
        windows = windows.swapaxes(0, 1)
        # (batch_size, n_features, window_size)

        return windows

    def _get_initial_z(self, p_0_chains_u, p_0_chains_l, z_persistent, batch_size):
        p_0_z_u = self.sample_gaussian(n_dim=self.z_size_up, n_samples=batch_size)
        p_0_z_u = p_0_z_u.to(self.device)

        p_0_z_l = self.sample_gaussian(
            n_dim=self.z_size, n_samples=batch_size * self.a_L
        )
        p_0_z_l = p_0_z_l.to(self.device)

        p_0_z = [p_0_z_u, p_0_z_l]

        return p_0_z

    def forward(self, Y, mask, **kwargs):
        Y = Y.clone()
        mask = mask.clone()
        if mask.ndim == 2:
            mask = mask.unsqueeze(1).repeat(1, self.n_features, 1)

        batch_size, _, _ = Y.shape

        # Normalize windows
        x_scales = Y[:, :, [0]]
        if self.normalize_windows:
            Y = Y - x_scales
            x_scales = x_scales.to(self.device)

        # Hide with mask
        Y = Y * mask

        # Gaussian noise, not used if generator is in eval mode
        if self.model.train:
            Y = Y + self.noise_std * (torch.randn(Y.shape).to(self.device))

        Y = self._preprocess_batch_hierarchy(windows=Y)
        mask = self._preprocess_batch_hierarchy(windows=mask)

        z_0 = self._get_initial_z(
            p_0_chains_l=self.p_0_chains_l,
            p_0_chains_u=self.p_0_chains_u,
            z_persistent=self.z_persistent,
            batch_size=batch_size,
        )

        # Sample z with Langevin Dynamics
        z, _, _ = self.infer_z(
            z=z_0, Y=Y, mask=mask, n_iters=self.z_iters, with_noise=self.z_with_noise
        )
        Y_hat = self.model(input=z, mask=mask)

        Y = self._postprocess_batch_hierarchy(windows=Y)
        mask = self._postprocess_batch_hierarchy(windows=mask)
        Y_hat = self._postprocess_batch_hierarchy(windows=Y_hat)

        if self.normalize_windows:
            Y = Y + x_scales
            Y_hat = Y_hat + x_scales
            Y = Y * mask
            Y_hat = Y_hat * mask

        return DGHLOutputs(timeseries=Y, reconstruction=Y_hat, mask=mask)

    def compute_loss(self, Y, Y_hat):
        # Loss
        loss = 0.5 * self.mse_loss(Y, Y_hat)
        return loss

    def training_step(self, x_enc, input_mask, **kwargs):
        self.model.train()

        outputs = self.forward(x_enc, input_mask, **kwargs)
        loss = self.compute_loss(outputs.timeseries, outputs.reconstruction)

        return DGHLOutputs(
            timeseries=outputs.timeseries,
            reconstruction=outputs.reconstruction,
            mask=outputs.mask,
            loss=loss,
        )

    def eval_step(self, x_enc, input_mask, **kwargs):
        self.model.eval()
        outputs = self.training_step(x_enc, input_mask, **kwargs)
        return outputs

    def reconstruct(self, x_enc, input_mask, **kwargs):
        self.model.eval()

        Y = x_enc.clone()
        mask = input_mask.clone()

        if mask.ndim == 2:
            mask = mask.unsqueeze(1).repeat(1, self.n_features, 1)

        batch_size, _, _ = Y.shape

        # Normalize windows
        x_scales = Y[:, :, [0]]
        if self.normalize_windows:
            Y = Y - x_scales
            x_scales = x_scales.to(self.device)

        # Hide with mask
        Y = Y * mask

        Y = self._preprocess_batch_hierarchy(windows=Y)
        mask = self._preprocess_batch_hierarchy(windows=mask)

        z_0 = self._get_initial_z(
            p_0_chains_l=self.p_0_chains_l,
            p_0_chains_u=self.p_0_chains_u,
            z_persistent=self.z_persistent,
            batch_size=batch_size,
        )

        # Sample z with Langevin Dynamics
        z, _, _ = self.infer_z(
            z=z_0, Y=Y, mask=mask, n_iters=self.z_iters_inference, with_noise=False
        )

        mask = torch.ones(mask.shape).to(
            self.device
        )  # In forward of generator, mask is all ones to reconstruct everything
        Y_hat = self.model(input=z, mask=mask)

        mask = self._postprocess_batch_hierarchy(windows=mask)
        Y_hat = self._postprocess_batch_hierarchy(windows=Y_hat)

        if self.normalize_windows:
            Y_hat = Y_hat + x_scales
            Y_hat = Y_hat * mask

        return DGHLOutputs(reconstruction=Y_hat, mask=mask)


class Generator(nn.Module):
    def __init__(
        self,
        window_size=32,
        hidden_multiplier=32,
        latent_size=100,
        n_features=3,
        max_filters=256,
        kernel_multiplier=1,
    ):
        super(Generator, self).__init__()

        n_layers = int(np.log2(window_size))
        layers = []
        filters_list = []
        # First layer
        filters = min(max_filters, hidden_multiplier * (2 ** (n_layers - 2)))
        layers.append(
            nn.ConvTranspose1d(
                in_channels=latent_size,
                out_channels=filters,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm1d(filters))
        filters_list.append(filters)
        # Hidden layers
        for i in reversed(range(1, n_layers - 1)):
            filters = min(max_filters, hidden_multiplier * (2 ** (i - 1)))
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=filters_list[-1],
                    out_channels=filters,
                    kernel_size=4 * kernel_multiplier,
                    stride=2,
                    padding=1 + (kernel_multiplier - 1) * 2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            filters_list.append(filters)

        # Output layer
        layers.append(
            nn.ConvTranspose1d(
                in_channels=filters_list[-1],
                out_channels=n_features,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, input, mask=None):
        input = self.layers(input)

        # Hide mask
        if mask is not None:
            input = input * mask

        return input
