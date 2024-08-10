"""Heteroscedastic classification NN implementation as a wrapper class. The dropout
layout is based on https://github.com/google/uncertainty-baselines and the method is
based on https://arxiv.org/abs/1703.04977."""

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


class HetClassNNWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates a Dropout model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_probability: float,
        is_filterwise_dropout: bool,
        num_mc_samples: int,
        num_integral_mc_samples: int,  # TODO: wire up
    ):
        super().__init__(model)

        self.num_mc_samples = num_mc_samples
        self.num_integral_mc_samples = num_integral_mc_samples

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

        self.reset_classifier(model.num_classes)
        self.eps = 1e-10

    def forward(self, inputs):
        if self.training:
            return self.predict_single(inputs)

        sampled_features = []
        sampled_logits = []
        sampled_internal_logits = []
        for _ in range(self.num_mc_samples):
            # features: [B, D]
            # internal_logits: [B, C]
            # logit_mc_samples: [B, S', C]
            features, internal_logits, logit_mc_samples = self.predict_single(
                inputs=inputs, return_bundle=True
            )
            logits = (
                F.softmax(logit_mc_samples, dim=-1).mean(dim=1).add(self.eps).log()
            )  # [B, C]

            sampled_features.append(features)
            sampled_logits.append(logits)
            sampled_internal_logits.append(internal_logits)

        sampled_features = torch.stack(sampled_features, dim=1)  # [B, S, D]
        mean_features = sampled_features.mean(dim=1)  # [B, D]
        sampled_logits = torch.stack(sampled_logits, dim=1)  # [B, S, C]
        sampled_internal_logits = torch.stack(
            sampled_internal_logits, dim=1
        )  # [B, S, C]

        return {
            "logit": sampled_logits,
            "internal_logit": sampled_internal_logits,
            "feature": mean_features,
        }

    def predict_single(self, inputs, return_bundle=False):
        pre_logits = self.model.forward_head(
            self.model.forward_features(inputs), pre_logits=True
        )  # [B, C]
        logits = self.model.fc(
            pre_logits
        )  # [B, C] TODO: breaks for ViTs (self.model.head)
        variances = self.log_var(pre_logits).exp()  # [B, C]
        stds = variances.sqrt()  # [B, C]

        logit_mc_samples = logits.unsqueeze(1) + stds.unsqueeze(1) * torch.randn(
            inputs.shape[0], self.num_integral_mc_samples, self.model.num_classes
        )  # [B, S', C]

        if return_bundle:
            return pre_logits, logits, logit_mc_samples
        else:
            return logit_mc_samples

    def reset_classifier(self, num_classes, *args, **kwargs):
        self.log_var = nn.Linear(
            in_features=self.num_features, out_features=num_classes
        )
        self.model.reset_classifier(num_classes, *args, **kwargs)

    def forward_features(self, inputs):
        raise ValueError(f"forward_features cannot be called directly for {type(self)}")

    def forward_head(self, features):
        raise ValueError(f"forward_head cannot be called directly for {type(self)}")
