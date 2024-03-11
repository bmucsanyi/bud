"""DDU implementation as a wrapper class.

Implementation based on https://github.com/omegafragger/DDU
"""

from functools import partial

import torch
from torch import nn
import warnings

from bud.utils.replace import replace
from bud.wrappers.temperature_wrapper import TemperatureWrapper
from bud.utils.metrics import centered_cov
from bud.wrappers.sngp_wrapper import SpectralNormalizedConv2dDUE


DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-308, 0, 1)]


class DDUWrapper(TemperatureWrapper):
    """
    This module takes a model as input and creates an SNGP from it.
    """

    def __init__(
        self,
        model: nn.Module,
        is_spectral_normalized: bool,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
    ):
        super().__init__(model, None)

        self.classwise_probs = None
        self.gmm = None

        if is_spectral_normalized:
            SNC = partial(
                SpectralNormalizedConv2dDUE,
                spectral_normalization_iteration,
                spectral_normalization_bound,
            )
            replace(model, "Conv2d", SNC)

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)
        logits = logits / self.temperature

        if self.training:
            return logits
        else:
            if self.gmm is None:
                warnings.warn("GMM has not been fit yet; giving constant EU estimates.")

                gmm_log_density = torch.ones((x.shape[0],))
            else:
                gmm_densities = self.gmm.log_prob(features[:, None, :]).exp()  # [B, C]
                gmm_log_density = (
                    (gmm_densities * self.classwise_probs)
                    .sum(dim=1)
                    .log()
                    .clamp(min=1e-10)
                )

            return {
                "logit": logits,
                "feature": features,
                "gmm_neg_log_density": -gmm_log_density,
            }

    def fit_gmm(self, train_loader, max_num_training_samples):
        features, labels = self._get_features(
            train_loader=train_loader, max_num_training_samples=max_num_training_samples
        )
        self._get_gmm(features=features, labels=labels)

    def _get_features(self, train_loader, max_num_training_samples):
        num_samples = min(len(train_loader.dataset), max_num_training_samples)
        features = torch.empty((num_samples, self.num_features))
        labels = torch.empty((num_samples,), dtype=torch.int)
        device = next(self.model.parameters()).device

        with torch.no_grad():
            current_num_samples = 0
            for input, label in train_loader:
                if current_num_samples + input.shape[0] > num_samples:
                    overhead = current_num_samples + input.shape[0] - num_samples
                    modified_batch_size = input.shape[0] - overhead
                    input = input[:modified_batch_size]

                input = input.to(device)

                feature = self.model.forward_head(
                    self.model.forward_features(input), pre_logits=True
                )

                start, end = current_num_samples, current_num_samples + input.shape[0]

                features[start:end] = feature.detach().cpu()
                labels[start:end] = label.detach().cpu()

                current_num_samples += input.shape[0]
                if current_num_samples >= num_samples:
                    break

        return features, labels

    def _get_gmm(self, features, labels):
        num_classes = self.model.num_classes
        classwise_probs = (
            torch.bincount(labels, minlength=num_classes).float() / labels.shape[0]
        )

        classwise_mean_features = torch.stack(
            [torch.mean(features[labels == c], dim=0) for c in range(num_classes)]
        )

        classwise_cov_features = torch.stack(
            [
                centered_cov(features[labels == c] - classwise_mean_features[c])
                for c in range(num_classes)
            ]
        )

        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1],
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features,
                    covariance_matrix=classwise_cov_features + jitter,
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
            break

        self.classwise_probs = classwise_probs
        self.gmm = gmm
