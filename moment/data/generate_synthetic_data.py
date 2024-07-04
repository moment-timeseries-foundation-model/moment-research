from typing import Tuple

import torch
from torch import nn

from moment.utils.utils import control_randomness


class SyntheticDataset(nn.Module):
    def __init__(
        self,
        n_samples: int = 1024,
        seq_len: int = 512,
        freq: int = 1,
        freq_range: Tuple[int, int] = (1, 32),
        amplitude_range: Tuple[int, int] = (1, 32),
        trend_range: Tuple[int, int] = (1, 32),
        baseline_range: Tuple[int, int] = (1, 32),
        noise_mean: float = 0.0,
        noise_std: float = 0.1,
        random_seed: int = 13,
        **kwargs,
    ):
        super(SyntheticDataset, self).__init__()

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.freq = freq
        self.freq_range = freq_range
        self.amplitude_range = amplitude_range
        self.trend_range = trend_range
        self.baseline_range = baseline_range
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.random_seed = random_seed
        control_randomness(self.random_seed)

    def __repr__(self):
        return (
            f"SyntheticDataset(n_samples={self.n_samples},"
            + f"seq_len={self.seq_len},"
            + f"freq={self.freq},"
            + f"freq_range={self.freq_range},"
            + f"amplitude_range={self.amplitude_range},"
            + f"trend_range={self.trend_range},"
            + f"baseline_range={self.baseline_range},"
            + f"noise_mean={self.noise_mean},"
            + f"noise_std={self.noise_std},"
            + f"random_seed={self.random_seed})"
        )

    def _generate_noise(self):
        epsilon = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=(self.n_samples, self.seq_len),
        )

        return epsilon

    def _generate_x(self):
        t = (
            torch.linspace(start=0, end=1, steps=self.seq_len)
            .unsqueeze(0)
            .repeat(self.n_samples, 1)
        )
        x = 2 * self.freq * torch.pi * t
        return x, t

    def gen_sinusoids_with_varying_freq(self):
        c = (
            torch.linspace(
                start=self.freq_range[0], end=self.freq_range[1], steps=self.n_samples
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(c * x) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_correlation(self):
        c = (
            torch.linspace(start=0, end=2 * np.pi, steps=self.n_samples)
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x + c) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_amplitude(self):
        c = (
            torch.linspace(
                start=self.amplitude_range[0],
                end=self.amplitude_range[1],
                steps=self.n_samples,
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )

        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = c * torch.sin(x) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_trend(self):
        c = (
            torch.linspace(
                start=self.trend_range[0], end=self.trend_range[1], steps=self.n_samples
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        # c = torch.cat((c, c), dim=0)
        # directions = torch.ones(self.n_samples, self.seq_len)
        # directions[self.n_samples//2:, :] = -1
        x, t = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x) + t**c + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_baseline(self):
        c = (
            torch.linspace(
                start=self.baseline_range[0],
                end=self.baseline_range[1],
                steps=self.n_samples,
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x) + c + epsilon
        y = y.unsqueeze(1)

        return y, c


"""
Mackey, M. C. and Glass, L. (1977). Oscillation and chaos in physiological control systems.
Science, 197(4300):287-289.

dy/dt = -by(t)+ cy(t - tau) / 1+y(t-tau)^10
"""

import numpy as np


def get_data(
    b: float = 0.1,
    c: float = 0.2,
    tau: float = 17,
    initial_values: np.ndarray = np.linspace(0.5, 1.5, 18),
    iterations: int = 1000,
) -> list:
    """
    Return a list with the Mackey-Glass chaotic time series.

    :param b: Equation coefficient
    :param c: Equation coefficient
    :param tau: Lag parameter, default: 17
    :param initial_values: numpy array with the initial values of y. Default: np.linspace(0.5,1.5,18)
    :param iterations: number of iterations. Default: 1000
    :return:
    """
    y = initial_values.tolist()

    for n in np.arange(len(y) - 1, iterations + 100):
        y.append(y[n] - b * y[n] + c * y[n - tau] / (1 + y[n - tau] ** 10))

    return y[100:]
