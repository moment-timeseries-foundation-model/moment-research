import logging

import torch
import torch.nn as nn

from moment.utils.utils import get_huggingface_model_dimensions

from .base import BaseModel
from .moment import MOMENT

T5_FAMILY = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]

GPT2_FAMILY = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


class FrozenPretrainedTransformer(BaseModel):
    def __init__(self, configs, **kwargs):
        super(FrozenPretrainedTransformer, self).__init__()
        """
        Parameters
        ----------
        return_last_only: bool
            Whether to take final hidden state of tokens 
            corresponding to last patch. True by default.
        
        use_embeddings_for_input: bool
            Whether to use embeddings for input. False by default.
        
        """

        self.configs = configs
        self.input_dim = configs.input_dim
        self.output_dim = configs.output_dim

        self.return_last_only = configs.return_last_only
        self.use_embeddings_for_input = configs.use_embeddings_for_input
        self.orth_gain = configs.orth_gain
        self.randomly_initialize_backbone = configs.randomly_initialize_backbone
        self.transformer_type = configs.transformer_type
        self.transformer_backbone = configs.transformer_backbone
        self.enable_gradient_checkpointing = configs.enable_gradient_checkpointing

        # When loading pre-trained MOMENT model
        self.model_name = configs.model_name
        self.pretraining_run_name = (
            configs.pretraining_run_name
        )  # This will be None for huggingface models.
        self.pretraining_opt_steps = (
            configs.pretraining_opt_steps
        )  # This will be None for huggingface models.

        # Frozen pre-trained transformer parameters
        self.freeze_layer_norm = configs.freeze_layer_norm
        self.freeze_pos = configs.freeze_pos
        self.freeze_ff = configs.freeze_ff
        self.freeze_attn = configs.freeze_attn

        self.input_layer_sizes = (
            [] if configs.input_layer_sizes is None else configs.input_layer_sizes
        )
        self.output_layer_sizes = (
            [] if configs.output_layer_sizes is None else configs.output_layer_sizes
        )
        self.dropout = configs.dropout
        self.d_model = (
            configs.d_model
        )  # There should be a better way to get this. Currently needs to be defined.

        self.d_model = None
        self.sequence_model = self._get_sequence_model()

        if self.use_embeddings_for_input:
            print("Using embeddings for input")
            self.input_net = nn.Embedding(self.input_dim, self.d_model)
        else:
            print("Using linear layers for input")
            self.input_net = self._get_input_net()

        self.output_net = self._get_output_net()

        if configs.freeze_transformer_backbone:
            self._freeze_transformer_backbone()

    def _get_sequence_model(self):
        if self.model_name == "MOMENT":
            print("========== Loading MOMENT model ==========")
            self.d_model = get_huggingface_model_dimensions(self.transformer_backbone)

            # Load pre-trained weights
            checkpoint = BaseModel.load_pretrained_weights(
                run_name=self.configs.pretraining_run_name, opt_steps=None
            )
            pretrained_model = MOMENT(configs=self.configs)
            pretrained_model.load_state_dict(checkpoint["model_state_dict"])

            return pretrained_model.encoder  # We only return the backbone transformer

        elif self.model_name in T5_FAMILY + GPT2_FAMILY:
            if self.randomly_initialize_backbone:
                logging.warning(f"Randomly initializing: {self.model_name}")

            if self.model_name in T5_FAMILY:
                assert self.transformer_type == "encoder_only"

            return self._get_huggingface_transformer()

        else:
            raise ValueError(f"Model name {self.model_name} not supported.")

    def _get_flant5_models(self):
        from transformers import T5Config, T5EncoderModel, T5Model

        print("========== Loading FLAN-T5 model ==========")

        self.d_model = get_huggingface_model_dimensions(self.transformer_backbone)

        if self.randomly_initialize_backbone:
            model_config = T5Config.from_pretrained(self.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(f"Initializing randomly initialized\
                          transformer from {self.transformer_backbone}.")
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                self.transformer_backbone
            )
            logging.info(f"Initializing pre-trained \
                          transformer from {self.transformer_backbone}.")

        if self.transformer_type == "encoder_only":
            transformer_backbone = transformer_backbone.get_encoder()
        elif self.transformer_type == "decoder_only":
            transformer_backbone = transformer_backbone.get_decoder()

        if self.enable_gradient_checkpointing:
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def _get_gpt2_models(self):
        from transformers import GPT2Config, GPT2Model

        print("========== Loading GPT-2 model ==========")

        self.d_model = GPT2Config.from_pretrained(self.transformer_backbone).n_embd

        if self.randomly_initialize_backbone:
            model_config = GPT2Config.from_pretrained(self.transformer_backbone)
            transformer_backbone = GPT2Model(model_config)
            logging.info(f"Initializing randomly initialized\
                          transformer from {self.transformer_backbone}.")
        else:
            transformer_backbone = GPT2Model.from_pretrained(self.transformer_backbone)
            logging.info(f"Initializing pre-trained \
                          transformer from {self.transformer_backbone}.")

        if self.enable_gradient_checkpointing:
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def _get_huggingface_transformer(self):
        if self.transformer_backbone in T5_FAMILY:
            return self._get_flant5_models()
        elif self.transformer_backbone in GPT2_FAMILY:
            return self._get_gpt2_models()

    def _get_input_net(self):
        input_layers = []
        last_output_size = self.input_dim

        for size in self.input_layer_sizes:
            layer = nn.Linear(last_output_size, size)
            if self.orth_gain is not None:
                torch.nn.init.orthogonal_(layer.weight, gain=self.orth_gain)
            layer.bias.data.zero_()

            input_layers.append(layer)
            input_layers.append(nn.ReLU())
            input_layers.append(nn.Dropout(self.dropout))
            last_output_size = size

        final_linear = nn.Linear(last_output_size, self.d_model)
        if self.orth_gain is not None:
            torch.nn.init.orthogonal_(final_linear.weight, gain=self.orth_gain)
        final_linear.bias.data.zero_()

        input_layers.append(final_linear)
        input_layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*input_layers)

    def _get_output_net(self):
        output_layers = []
        last_output_size = self.d_model
        for size in self.output_layer_sizes:
            output_layers.append(nn.Linear(last_output_size, size))
            output_layers.append(nn.ReLU())
            output_layers.append(nn.Dropout(self.dropout))
            last_output_size = size
        output_layers.append(nn.Linear(last_output_size, self.output_dim))

        return nn.Sequential(*output_layers)

    def _freeze_transformer_backbone(self):
        for name, p in self.sequence_model.named_parameters():
            name = name.lower()
            if "ln" in name or "norm" in name or "layer_norm" in name:
                p.requires_grad = not self.freeze_layer_norm
            elif "wpe" in name or "position_embeddings" in name or "pos_drop" in name:
                p.requires_grad = not self.freeze_pos
            elif "mlp" in name or "densereludense" in name:
                p.requires_grad = not self.freeze_ff
            elif "attn" in name or "selfattention" in name:
                p.requires_grad = not self.freeze_attn
            else:
                p.requires_grad = False

    def forward(self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, **kwargs):
        """
        x_enc : [batch_size x original_dim x seq_len]
        input_mask :
        """

        if len(x_enc.shape) == 2:  # NLP
            batch_size, original_dim = x_enc.shape
            ratio = 1
        else:
            batch_size, seq_len, original_dim = x_enc.shape
            if original_dim != self.input_dim and not self.use_embeddings_for_input:
                if original_dim % self.input_dim != 0:
                    raise ValueError(
                        f"dimension of x must be divisible by patch size. "
                        f"Orig dim: {original_dim}, input dim: {self.input_dim}"
                    )
                ratio = original_dim // self.input_dim
                x_enc = x_enc.reshape(batch_size, -1, self.input_dim)
                # [batch_size, seq_len*num_patches, patch_dim]
            else:
                ratio = 1

        enc_in = self.input_net(x_enc)

        outputs = self.sequence_model(inputs_embeds=enc_in, return_dict=True)
        enc_out = outputs.last_hidden_state
        # NOTE: This is different from MOMENT. No attention mask!

        # Take final hidden state of tokens corresponding to last patch
        if self.return_last_only:
            enc_out = enc_out[:, -ratio:]

        # Single linear layer applied to last hidden state
        dec_out = self.output_net(enc_out)

        if len(x_enc.shape) == 2:  # NLP
            dec_out = torch.squeeze(dec_out, dim=1)

        # If we did patch resizing above, return in the original shape [batch_size, original_dim, seq_len]
        if self.return_last_only and ratio > 1:
            _, n_tokens, _ = dec_out.shape
            dec_out = dec_out.reshape(
                batch_size, n_tokens // ratio, ratio * self.output_dim
            )

        return dec_out
