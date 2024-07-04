import logging
import warnings

import torch
from torch import nn
from wandb import AlertLevel

from moment.common import TASKS
from moment.data.base import TimeseriesOutputs
from moment.utils.masking import Masking

from .layers.embed import PatchEmbedding, Patching
from .layers.revin import RevIN


class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x seq_len], where seq_len = n_patches * patch_len
        """

        # x = x.transpose(2, 3)                 # [batch_size x n_channels x n_patches x d_model]
        x = self.linear(
            self.dropout(x)
        )  # [batch_size x n_channels x n_patches x patch_len]
        x = x.flatten(start_dim=2, end_dim=3)  # [batch_size x n_patches x seq_len]
        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_channels * d_model, n_classes)

    def forward(self, x, input_mask: torch.Tensor = None):
        """
        x: [batch_size x n_channels x d_model x n_patches]
        output: [batch_size x n_classes]
        """
        # NOTE: Support for input mask is not yet implemented

        x = x.nanmean(
            dim=-1
        ).squeeze()  # x: batch_size x n_channels x n_patches x d_model
        # x = x[:,:,:,-1]             # x: batch_size x n_channels x d_model
        x = self.flatten(x)  # x: batch_size x n_channels * d_model
        y = self.linear(self.dropout(x))
        # y: batch_size x n_classes
        return y


class ForecastingHead(nn.Module):
    def __init__(
        self,
        head_nf: int = 768 * 64,
        forecast_horizon: int = 96,
        head_dropout: int = 0.1,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        """
        x: [batch_size x n_channels x d_model x n_patches]
        output: [batch_size x n_channels x forecast_horizon]
        """

        # x: batch_size x n_channels x n_patches x d_model
        x = self.flatten(x)
        # x: batch_size x n_channels x n_patches*d_model
        x = self.linear(x)
        # x: batch_size x n_channels x forecast_horizon
        x = self.dropout(x)
        return x


class MOMENT(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        configs = self._validate_inputs(configs)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        padding = configs.patch_stride_len

        # Normalization, patching and embedding
        self.normalizer = RevIN(num_features=1, affine=configs.revin_affine)
        self.tokenizer = Patching(
            patch_len=configs.patch_len, stride=configs.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            configs.patch_len,
            configs.patch_stride_len,
            padding,
            configs.dropout,
            configs.orth_gain,
        ).to(configs.device)
        self.mask_generator = Masking(mask_ratio=configs.mask_ratio)

        # Encoder
        self.encoder = self.get_patchtst_encoder(configs)

        # Prediction Head
        self.head_nf = configs.d_model * (
            (max(configs.seq_len, configs.patch_len) - configs.patch_len)
            // configs.patch_stride_len
            + 1
        )
        if self.task_name == TASKS.PRETRAINING:
            self.head = PretrainHead(
                configs.d_model, configs.patch_len, configs.dropout, configs.orth_gain
            )
        elif self.task_name == TASKS.CLASSIFICATION:
            self.head = ClassificationHead(
                configs.n_channels, configs.d_model, configs.num_class, configs.dropout
            )
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")

    def _validate_inputs(self, configs):
        if configs.use_pretrained_model:
            raise ValueError("Pretrained model not yet supported.")

        if configs.d_model is None:
            raise ValueError("d_model must be specified.")

        if configs.patch_stride_len != configs.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return configs

    def freeze_encoder(self, configs):
        if configs.freeze_encoder:
            for name, p in self.encoder.named_parameters():
                name = name.lower()
                if "ln" in name or "norm" in name or "layer_norm" in name:
                    p.requires_grad = not configs.freeze_layer_norm
                elif (
                    "wpe" in name or "position_embeddings" in name or "pos_drop" in name
                ):
                    p.requires_grad = not configs.freeze_pos
                elif "mlp" in name or "densereludense" in name:
                    p.requires_grad = not configs.freeze_ff
                elif "attn" in name or "selfattention" in name:
                    p.requires_grad = not configs.freeze_attn
                else:
                    p.requires_grad = False

    def get_patchtst_encoder(self, configs):
        from .layers.self_attention_family import AttentionLayer, FullAttention
        from .layers.transformer_encoder_decoder import Encoder, EncoderLayer

        encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=configs.attention_dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        return encoder

    def pretraining(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
        mask  : [batch_size x seq_len]
        input_mask : [batch_size x seq_len]
        """
        batch_size, n_channels, _ = x_enc.shape
        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

        # Normalization
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # x_enc = self.normalizer(x=x_enc, mask=input_mask, mode='norm')
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        # [batch_size x n_channels x seq_len]
        x_enc = self.tokenizer(x=x_enc)
        # [batch_size x n_channels x n_patches x patch_len]

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size x n_channels x n_patches x d_model]

        # Encoder
        enc_out = self.encoder(inputs_embeds=enc_in)

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        if self.configs.check_illegal_model_weights:
            illegal_output = self._check_model_weights_for_illegal_values()
        else:
            illegal_output = None

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=mask,
            illegal_output=illegal_output,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        if self.task_name == TASKS.PRETRAINING:
            return self.pretraining(
                x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs
            )
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
        return
