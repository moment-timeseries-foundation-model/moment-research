from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from moment.common import TASKS
from moment.data.base import TimeseriesOutputs
from moment.models.base import BaseModel
from moment.utils.utils import get_anomaly_criterion

from .layers.embed import DataEmbedding
from .layers.self_attention_family import AnomalyAttention, AttentionLayer


@dataclass
class AnomalyTransformerOutputs:
    reconstruction: torch.Tensor = None
    series: torch.Tensor = None
    prior: torch.Tensor = None
    sigmas: torch.Tensor = None


class AnomalyTransformer(BaseModel):
    """
    Anomaly transformer: Time series anomaly detection with association discrepancy
    References
    -----------
    [1] Xu, Jiehui, et al. "Anomaly transformer: Time series anomaly detection with association discrepancy." (2021).
        https://arxiv.org/abs/2110.02642
    """

    def __init__(self, configs):
        super(AnomalyTransformer, self).__init__()

        self.task_name = configs.task_name
        self.win_size = configs.seq_len
        self.configs = configs

        self.enc_in = configs.n_channels
        self.c_out = configs.n_channels
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = configs.activation

        self._build_model()

    def _build_model(self):
        # Encoding
        self.embedding = DataEmbedding(
            c_in=self.enc_in,
            d_model=self.d_model,
            dropout=self.dropout,
            model_name=self.configs.model_name,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            self.win_size, False, attention_dropout=self.dropout
                        ),
                        self.d_model,
                        self.n_heads,
                        anomaly_attention=True,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )

        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def reconstruct(self, x_enc, **kwargs):
        # x_enc: batch_size x n_channels x seq_len
        enc_out = self.embedding(x_enc)
        # batch_size x seq_len x d_model
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        # batch_size x seq_len x d_model
        enc_out = self.projection(enc_out)
        # batch_size x seq_len x n_channels
        enc_out = enc_out.permute(0, 2, 1)
        # batch_size x n_channels x seq_len

        return AnomalyTransformerOutputs(
            reconstruction=enc_out, series=series, prior=prior, sigmas=sigmas
        )

    def detect_anomalies(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        anomaly_criterion: str = "mse",
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        anomaly_criterion : str
        """
        outputs = self.reconstruct(x_enc=x_enc)
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=outputs.reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )

    def forward(self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, **kwargs):
        if self.task_name == TASKS.ANOMALY_DETECTION:
            return self.detect_anomalies(x_enc=x_enc, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model: int = 512,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(EncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list
