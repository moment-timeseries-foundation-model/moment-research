import torch
import torch.nn as nn


class MovingAverageBlock(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAverageBlock, self).__init__()
        self.kernel_size = kernel_size
        self.average = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time-series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.average(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecompositionBlock(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecompositionBlock, self).__init__()
        self.moving_average = MovingAverageBlock(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_average(x)
        residual = x - moving_mean
        return residual, moving_mean


class MultipleSeriesDecompositionBlock(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(MultipleSeriesDecompositionBlock, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [
            SeriesDecompositionBlock(kernel_size=kernel) for kernel in kernel_size
        ]

    def forward(self, x):
        moving_mean = []
        residual = []
        for func in self.series_decomp:
            seasonality, moving_avg = func(x)
            moving_mean.append(moving_avg)
            residual.append(seasonality)

        seasonality = sum(residual) / len(residual)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return seasonality, moving_mean
