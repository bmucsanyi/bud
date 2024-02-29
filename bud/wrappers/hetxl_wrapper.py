"""HET-XL implementation as a wrapper class. Heteroscedastic Gaussian sampling based on
https://github.com/google/uncertainty-baselines"""

import torch
import torch.nn.functional as F
from torch import nn

from bud.wrappers.model_wrapper import PosteriorWrapper


class HETXLHead(nn.Module):
    def __init__(
        self, matrix_rank, num_mc_samples, num_features, temperature, classifier
    ):
        super().__init__()
        self.matrix_rank = matrix_rank
        self.num_mc_samples = num_mc_samples
        self.num_features = num_features

        self.low_rank_cov_layer = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_features * self.matrix_rank,
        )
        self.diagonal_std_layer = nn.Linear(
            in_features=self.num_features, out_features=self.num_features
        )
        self.min_scale_monte_carlo = 1e-3

        self.temperature = temperature
        self.classifier = classifier

    def forward(self, features):
        # Shape variables
        B, D = features.shape
        R = self.matrix_rank
        S = self.num_mc_samples

        low_rank_cov = self.low_rank_cov_layer(features).reshape(-1, D, R)  # [B, D, R]
        diagonal_std = (
            F.softplus(self.diagonal_std_layer(features)) + self.min_scale_monte_carlo
        )  # [B, D]

        # TODO: https://github.com/google/edward2/blob/main/edward2/jax/nn/heteroscedastic_lib.py#L189
        diagonal_samples = diagonal_std.unsqueeze(1) * torch.randn(
            B, S, D, device=features.device
        )  # [B, S, D]
        standard_samples = torch.randn(B, S, R, device=features.device)  # [B, S, R]
        einsum_res = torch.einsum(
            "bdr,bsr->bsd", low_rank_cov, standard_samples
        )  # [B, S, D]
        samples = einsum_res + diagonal_samples  # [B, S, D]
        pre_logits = features.unsqueeze(1) + samples  # [B, S, D]
        logits = self.classifier(pre_logits)  # [B, S, C]
        logits_temperature = logits / self.temperature

        # TODO: https://github.com/google/edward2/blob/main/edward2/jax/nn/heteroscedastic_lib.py#L325

        return logits_temperature


class HETXLWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates a HET-XL model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        matrix_rank: int,
        num_mc_samples: int,
        temperature: float,
    ):
        super().__init__(model)

        self.matrix_rank = matrix_rank
        self.num_mc_samples = num_mc_samples
        self.temperature = temperature

        self.classifier = HETXLHead(
            matrix_rank=self.matrix_rank,
            num_mc_samples=self.num_mc_samples,
            num_features=self.num_features,
            classifier=self.model.get_classifier(),
            temperature=self.temperature,
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(
        self,
        matrix_rank: int = None,
        num_mc_samples: int = None,
        temperature: float = None,
        *args,
        **kwargs
    ):
        if matrix_rank is not None:
            self.matrix_rank = matrix_rank

        if num_mc_samples is not None:
            self.num_mc_samples = num_mc_samples

        if temperature is not None:
            self.temperature = temperature

        self.model.reset_classifier(*args, **kwargs)
        self.classifier = HETXLHead(
            matrix_rank=self.matrix_rank,
            num_mc_samples=self.num_mc_samples,
            num_features=self.num_features,
            classifier=self.model.get_classifier(),
            temperature=self.temperature,
        )
