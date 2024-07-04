from dataclasses import dataclass

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from moment.common import TASKS
from moment.models.layers.conv_blocks import InceptionBlockV1
from moment.models.layers.embed import DataEmbedding
from moment.utils.masking import Masking
from moment.utils.utils import get_anomaly_criterion


@dataclass
class TimesNetOutputs:
    backcast: torch.Tensor = None
    forecast: torch.Tensor = None
    timeseries: torch.Tensor = None
    reconstruction: torch.Tensor = None
    pretrain_mask: torch.Tensor = None
    mask: torch.Tensor = None
    anomaly_scores: torch.Tensor = None
    metadata: dict = None


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.forecast_horizon
        self.k = configs.top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            InceptionBlockV1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.n_channels
        self.c_out = configs.n_channels
        self.model_name = configs.model_name
        self.pred_len = configs.forecast_horizon

        self.model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )

        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout,
            model_name=self.configs.model_name,
        )

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.mask_generator = Masking(mask_ratio=configs.mask_ratio)

        if (
            self.task_name == TASKS.LONG_HORIZON_FORECASTING
            or self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        ):
            self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
            self.projection = nn.Linear(configs.d_model, self.c_out, bias=True)
        elif (
            self.task_name == TASKS.IMPUTATION
            or self.task_name == TASKS.ANOMALY_DETECTION
        ):
            self.projection = nn.Linear(configs.d_model, self.c_out, bias=True)
        elif self.task_name == TASKS.PRETRAINING:
            self.projection = nn.Linear(configs.d_model, self.c_out, bias=True)
        elif self.task_name == TASKS.CLASSIFICATION:
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")

    def forecast(self, x_enc, **kwargs):
        """
        Input:
            x_enc : [batch_size x n_channels x seq_len]
            mask : [batch_size x seq_len]
        Returns:
            forecast : [batch_size x n_channels x pred_len]
        """

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [batch_size x seq_len x n_channels]
        enc_out = enc_out.permute(0, 2, 1)  # [batch_size x n_channels x seq_len]
        enc_out = self.predict_linear(enc_out)  # Along temporal dimension
        enc_out = enc_out.permute(
            0, 2, 1
        )  # [batch_size x (seq_len+pred_len) x n_channels]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Project back
        dec_out = self.projection(enc_out)  # [batch_size x seq_len x n_channels]
        dec_out = dec_out.permute(
            0, 2, 1
        )  # [batch_size x n_channels x (seq_len+pred_len)]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev.repeat(1, 1, self.seq_len + self.pred_len))
        dec_out = dec_out + (means.repeat(1, 1, self.seq_len + self.pred_len))
        # [batch_size x n_channels x (seq_len+pred_len)]

        forecast = dec_out[:, :, -self.pred_len :]
        backcast = dec_out[:, :, : -self.pred_len]

        return TimesNetOutputs(backcast=backcast, forecast=forecast, timeseries=x_enc)

    def reconstruct(self, x_enc, mask=None, **kwargs):
        """
        Input:
            x_enc : [batch_size x n_channels x seq_len]
            mask : [batch_size x seq_len]
        Returns:
            forecast : [batch_size x n_channels x pred_len]
        """

        if mask is None:
            mask = torch.ones_like(x_enc)[:, 0, :]

        mask = mask.unsqueeze(1).repeat(
            1, x_enc.shape[1], 1
        )  # [batch_size x n_channels x seq_len]

        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=2) / torch.sum(mask == 1, dim=2)
        means = means.unsqueeze(2).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=2) / torch.sum(mask == 1, dim=2) + 1e-5
        )
        stdev = stdev.unsqueeze(2).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [batch_size x seq_len x n_channels]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Project back
        dec_out = self.projection(enc_out)  # [batch_size x seq_len x n_channels]
        dec_out = dec_out.permute(0, 2, 1)  # [batch_size x n_channels x seq_len]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev.repeat(1, 1, self.seq_len))
        dec_out = dec_out + (means.repeat(1, 1, self.seq_len))
        # [batch_size x n_channels x (seq_len+pred_len)]

        return TimesNetOutputs(
            reconstruction=dec_out, timeseries=x_enc, mask=mask[:, 0, :]
        )

    def pretraining(
        self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, **kwargs
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
        mask = mask.to(x_enc.device)
        mask = mask.unsqueeze(1).repeat(
            1, x_enc.shape[1], 1
        )  # [batch_size x n_channels x seq_len]

        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=2) / torch.sum(mask == 1, dim=2)
        means = means.unsqueeze(2).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=2) / torch.sum(mask == 1, dim=2) + 1e-5
        )
        stdev = stdev.unsqueeze(2).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [batch_size x seq_len x n_channels]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Project back
        dec_out = self.projection(enc_out)  # [batch_size x seq_len x n_channels]
        dec_out = dec_out.permute(0, 2, 1)  # [batch_size x n_channels x seq_len]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev.repeat(1, 1, self.seq_len))
        dec_out = dec_out + (means.repeat(1, 1, self.seq_len))
        # [batch_size x n_channels x (seq_len+pred_len)]

        return TimesNetOutputs(
            reconstruction=dec_out, timeseries=x_enc, pretrain_mask=mask[:, 0, :]
        )

    def detect_anomalies(
        self, x_enc: torch.Tensor, anomaly_criterion: str = "mse", **kwargs
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        anomaly_criterion : str
        """
        outputs = self.reconstruct(x_enc=x_enc)
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

        return TimesNetOutputs(
            reconstruction=outputs.reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )

    def classification(self, x_enc, **kwargs):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)

        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)

        return output

    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        if (
            self.task_name == TASKS.LONG_HORIZON_FORECASTING
            or self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        ):
            return self.forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.ANOMALY_DETECTION:
            return self.detect_anomalies(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.IMPUTATION:
            return self.reconstruct(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classification(x_enc=x_enc, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
