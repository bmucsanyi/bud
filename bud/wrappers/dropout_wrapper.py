"""Dropout implementation as a wrapper class. The dropout layout is based on
https://github.com/google/uncertainty-baselines"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from bud.utils.replace import replace
from bud.wrappers.model_wrapper import PosteriorWrapper


class ActivationDropout(nn.Module):
    def __init__(self, dropout_probability, is_filterwise_dropout, activation):
        super().__init__()
        self.activation = activation
        dropout_function = F.dropout2d if is_filterwise_dropout else F.dropout
        self.dropout = partial(dropout_function, p=dropout_probability, training=True)

    def forward(self, inputs):
        x = self.activation(inputs)
        x = self.dropout(x)
        return x


class DropoutWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates a Dropout model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_probability: float,
        is_filterwise_dropout: bool,
        num_mc_samples: int,
    ):
        super().__init__(model)

        self.num_mc_samples = num_mc_samples

        replace(
            model,
            "ReLU",
            partial(ActivationDropout, dropout_probability, is_filterwise_dropout),
        )
        replace(
            model,
            "GELU",
            partial(ActivationDropout, dropout_probability, is_filterwise_dropout),
        )

    def forward(self, inputs):
        if self.training:
            return self.model(inputs)  # [B, C]

        sampled_features = []
        sampled_logits = []
        for _ in range(self.num_mc_samples):
            features = self.model.forward_head(
                self.model.forward_features(inputs), pre_logits=True
            )
            logits = self.model(inputs)  # [B, C]

            sampled_features.append(features)
            sampled_logits.append(logits)

        sampled_features = torch.stack(sampled_features, dim=1)  # [B, S, D]
        mean_features = sampled_features.mean(dim=1)
        sampled_logits = torch.stack(sampled_logits, dim=1)  # [B, S, C]

        return {"logit": sampled_logits, "feature": mean_features}

    def forward_features(self, inputs):
        raise ValueError(f"forward_features cannot be called directly for {type(self)}")

    def forward_head(self, features):
        raise ValueError(f"forward_head cannot be called directly for {type(self)}")
