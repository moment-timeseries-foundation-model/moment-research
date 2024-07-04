import torch
import torch.nn as nn
import torch.nn.functional as F

from moment.common import TASKS
from moment.data.base import TimeseriesOutputs
from moment.models.base import BaseModel
from moment.models.layers.series_decomposition import SeriesDecompositionBlock
from moment.utils.utils import get_anomaly_criterion


class DLinear(BaseModel):
    """
    DLinear: Decomposition Linear Model
    References
    -----------
    [1] Zeng et al. Are Transformers Effective for Time Series Forecasting?
        https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.configs = configs
        self.share_model = configs.share_model
        self.n_channels = configs.n_channels

        if (
            self.task_name == TASKS.LONG_HORIZON_FORECASTING
            or self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        ):
            self.pred_len = configs.forecast_horizon
        else:
            self.pred_len = configs.seq_len

        # Series decomposition block from Autoformer
        self.decompsition = SeriesDecompositionBlock(kernel_size=configs.kernel_size)

        self._build_model()

    def _build_model(self):
        if not self.share_model:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()

            for i in range(self.n_channels):
                self.linear_seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.linear_trend.append(nn.Linear(self.seq_len, self.pred_len))

                self.linear_seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.linear_trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

            self.linear_seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.linear_trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

        if self.task_name == TASKS.CLASSIFICATION:
            self.act = F.gelu
            self.dropout = nn.Dropout(self.configs.head_dropout)
            self.projection = nn.Linear(
                self.configs.n_channels * self.configs.seq_len, self.configs.num_class
            )

    def encoder(self, x, **kwargs):
        seasonal_init, trend_init = self.decompsition(x)

        if not self.share_model:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.n_channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        x = seasonal_output + trend_output

        return x

    def forecast(self, x_enc, **kwargs):
        forecast = self.encoder(x_enc, **kwargs)
        return TimeseriesOutputs(forecast=forecast)

    def imputation(self, x_enc, **kwargs):
        reconstruction = self.encoder(x_enc, **kwargs)
        return TimeseriesOutputs(reconstruction=reconstruction)

    def detect_anomalies(
        self, x_enc: torch.Tensor, anomaly_criterion: str = "mse", **kwargs
    ):
        reconstruction = self.encoder(x_enc, **kwargs)
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, reconstruction)

        return TimeseriesOutputs(
            reconstruction=reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )

    def classification(self, x_enc, **kwargs):
        enc_out = self.encoder(x_enc, **kwargs)
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc: torch.Tensor, **kwargs):
        if (
            self.task_name == TASKS.SHORT_HORIZON_FORECASTING
            or self.task_name == TASKS.LONG_HORIZON_FORECASTING
        ):
            return self.forecast(x_enc=x_enc, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
