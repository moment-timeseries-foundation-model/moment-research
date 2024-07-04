from math import pi, sqrt

import numpy as np
import torch
import torch.nn as nn


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(
        self, win_size, device, mask_flag=True, scale=None, attention_dropout=0.0
    ):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.device = device
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size

        # self.distances = torch.zeros((window_size, window_size)).cuda()
        # for i in range(window_size):
        #     for j in range(window_size):
        #         self.distances[i][j] = abs(i - j)

        a = np.arange(window_size).repeat(window_size).reshape(window_size, window_size)
        b = (
            np.arange(window_size)
            .repeat(window_size)
            .reshape(window_size, window_size)
            .T
        )
        self.distances = torch.from_numpy(
            np.abs(a - b).astype(np.float32)
        )  # .to(self.device)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = (
            self.distances.unsqueeze(0)
            .unsqueeze(0)
            .repeat(sigma.shape[0], sigma.shape[1], 1, 1)
            .to(queries.device)
        )
        prior = 1.0 / (sqrt(2 * pi) * sigma) * torch.exp(-(prior**2) / 2 / (sigma**2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        return (V.contiguous(), series, prior, sigma)


class AttentionLayer(nn.Module):
    """Should be compatible with both AnomalyTransformer and vanilla transformer"""

    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        anomaly_attention: bool = False,
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.norm = nn.LayerNorm(d_model)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.anomaly_attention = anomaly_attention
        if self.anomaly_attention:
            self.sigma_projection = nn.Linear(d_model, n_heads)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.anomaly_attention:
            x = queries

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if self.anomaly_attention:
            sigma = self.sigma_projection(x).view(B, L, H)

        if self.anomaly_attention:
            out, series, prior, sigma = self.inner_attention(
                queries, keys, values, sigma, attn_mask
            )
        else:
            out, attn = self.inner_attention(
                queries, keys, values, attn_mask, tau=tau, delta=delta
            )
        out = out.view(B, L, -1)

        if self.anomaly_attention:
            return self.out_projection(out), series, prior, sigma
        else:
            return self.out_projection(out), series


# class AttentionLayer(nn.Module):
#     def __init__(self,
#                  attention,
#                  d_model,
#                  n_heads,
#                  d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn
