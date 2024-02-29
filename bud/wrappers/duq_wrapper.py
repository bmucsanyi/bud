"""DUQ implementation as a wrapper class.
Based on https://github.com/y0ast/deterministic-uncertainty-quantification"""

import torch
from torch import nn

from bud.wrappers.model_wrapper import SpecialWrapper


class DUQHead(nn.Module):
    def __init__(
        self,
        num_classes,
        num_features,
        rbf_length_scale,
        ema_momentum,
        num_hidden_features,
    ):
        super().__init__()
        self.num_classes = num_classes

        if num_hidden_features < 0:
            num_hidden_features = num_features

        self.num_hidden_features = num_hidden_features
        self.num_features = num_features
        self.rbf_length_scale = rbf_length_scale
        self.ema_momentum = ema_momentum

        self.weight = nn.Parameter(
            torch.empty(num_classes, num_hidden_features, num_features)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

        self.register_buffer(
            "ema_num_samples_per_class", torch.full((num_classes,), 128 / num_classes)
        )
        self.register_buffer(
            "ema_embedding_sums_per_class",
            torch.randn(num_classes, num_hidden_features),
        )

    def rbf(self, features):
        latent_features = torch.einsum(
            "clf,bf->bcl", self.weight, features
        )  # [B, C, L]

        centroids = (
            self.ema_embedding_sums_per_class
            / self.ema_num_samples_per_class.unsqueeze(dim=1)
        )  # [C, L]

        diffs = latent_features - centroids  # [B, C, L]
        rbf_values = (
            diffs.square()
            .mean(dim=-1)
            .div(2 * self.rbf_length_scale**2)
            .mul(-1)
            .exp()
        )  # [B, C]

        return rbf_values

    def update_centroids(self, features, targets):
        num_samples_per_class = targets.sum(dim=0)
        self.ema_num_samples_per_class = (
            self.ema_momentum * self.ema_num_samples_per_class
            + (1 - self.ema_momentum) * num_samples_per_class
        )

        latent_features = torch.einsum(
            "clf,bf->bcl", self.weight, features
        )  # [B, C, L]
        embedding_sums_per_class = torch.einsum(
            "bcl,bc->cl", latent_features, targets
        )  # [C, L]

        self.ema_embedding_sums_per_class = (
            self.ema_momentum * self.ema_embedding_sums_per_class
            + (1 - self.ema_momentum) * embedding_sums_per_class
        )

    def forward(self, features):
        rbf_values = self.rbf(features)

        if self.training:
            return rbf_values
        else:
            min_real = torch.finfo(rbf_values.dtype).min
            logit = (
                rbf_values.div(rbf_values.sum(dim=1, keepdim=True))
                .log()
                .clamp(min=min_real)
            )

            return {
                "logit": logit,  # [B, C]
                "feature": features,  # [B, D]
                "duq_value": 1 - rbf_values.max(dim=1)[0],  # [B]
                # TODO: ask Michael about .mul(-1) vs 1 / x
            }


class DUQWrapper(SpecialWrapper):
    """
    This module takes a model as input and creates a DUQ model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features,
        rbf_length_scale,
        ema_momentum,
    ):
        super().__init__(model)

        self.num_hidden_features = num_hidden_features
        self.rbf_length_scale = rbf_length_scale
        self.ema_momentum = ema_momentum

        self.classifier = DUQHead(
            num_classes=self.num_classes,
            num_features=self.num_features,
            rbf_length_scale=self.rbf_length_scale,
            ema_momentum=self.ema_momentum,
            num_hidden_features=self.num_hidden_features,
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(
        self,
        num_hidden_features: int = None,
        rbf_length_scale: float = None,
        ema_momentum: float = None,
        *args,
        **kwargs,
    ):
        if num_hidden_features is not None:
            self.num_hidden_features = num_hidden_features

        if rbf_length_scale is not None:
            self.rbf_length_scale = rbf_length_scale

        if ema_momentum is not None:
            self.ema_momentum = ema_momentum

        self.model.reset_classifier(*args, **kwargs)
        self.classifier = DUQHead(
            num_classes=self.num_classes,
            num_features=self.num_features,
            rbf_length_scale=self.rbf_length_scale,
            ema_momentum=self.ema_momentum,
            num_hidden_features=self.num_hidden_features,
        )

    def update_centroids(self, inputs, targets):
        features = self.model.forward_head(
            self.model.forward_features(inputs), pre_logits=True
        )
        self.classifier.update_centroids(features, targets)

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)

        return out
