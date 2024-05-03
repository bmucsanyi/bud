"""DDU implementation as a wrapper class.

Implementation based on https://github.com/omegafragger/DDU
"""

from functools import partial

import torch
from torch import nn
import warnings

from bud.utils.replace import register, replace, register_cond
from bud.wrappers.temperature_wrapper import TemperatureWrapper
from bud.utils.metrics import centered_cov
from bud.wrappers.sngp_wrapper import (
    Conv2dSpectralNormalizer,
    LinearSpectralNormalizer,
    SpectralNormalizedBatchNorm2d,
)


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
        is_batch_norm_spectral_normalized: bool,
        use_tight_norm_for_pointwise_convs: bool,
    ):
        super().__init__(model, None)

        self.register_buffer("classwise_probs", None)
        self.register_buffer("gmm_loc", None)
        self.register_buffer("gmm_covariance_matrix", None)
        self.gmm = None

        if is_spectral_normalized:
            LSN = partial(
                LinearSpectralNormalizer,
                spectral_normalization_iteration=spectral_normalization_iteration,
                spectral_normalization_bound=spectral_normalization_bound,
                dim=0,
                eps=1e-12,
            )

            CSN = partial(
                Conv2dSpectralNormalizer,
                spectral_normalization_iteration=spectral_normalization_iteration,
                spectral_normalization_bound=spectral_normalization_bound,
                eps=1e-12,
            )

            SNBN = partial(
                SpectralNormalizedBatchNorm2d,
                spectral_normalization_bound=spectral_normalization_bound,
            )

            if use_tight_norm_for_pointwise_convs:

                def is_pointwise_conv(conv2d: nn.Conv2d) -> bool:
                    return conv2d.kernel_size == (1, 1)

                register_cond(
                    model=model,
                    source_regex="Conv2d",
                    attribute_name="weight",
                    cond=is_pointwise_conv,
                    target_parametrization_true=LSN,
                    target_parametrization_false=CSN,
                )
            else:
                register(
                    model=model,
                    source_regex="Conv2d",
                    attribute_name="weight",
                    target_parametrization=CSN,
                )

            if is_batch_norm_spectral_normalized:
                replace(
                    model=model,
                    source_regex="BatchNorm2d",
                    target_module=SNBN,
                )

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
            if self.gmm_loc is None:
                warnings.warn("GMM has not been fit yet; giving constant EU estimates.")

                gmm_log_density = torch.ones((x.shape[0],))
            else:
                if self.gmm is None:
                    self.gmm = torch.distributions.MultivariateNormal(
                        loc=self.gmm_loc,
                        covariance_matrix=self.gmm_covariance_matrix,
                    )

                gmm_log_densities = self.gmm.log_prob(
                    features[:, None, :].cpu()
                ).cuda()  # [B, C]
                gmm_weighted_log_densities = (
                    gmm_log_densities + self.classwise_probs.log()
                )
                gmm_log_density = gmm_weighted_log_densities.logsumexp(dim=1)

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
        batch_size = train_loader.batch_size
        dataset_size = len(train_loader.dataset)
        drop_last = train_loader.drop_last

        # Calculate the effective number of samples based on drop_last setting
        if drop_last:
            total_full_batches = dataset_size // batch_size
            effective_dataset_size = total_full_batches * batch_size
        else:
            effective_dataset_size = dataset_size

        # Determine the actual number of samples to process
        num_samples = min(effective_dataset_size, max_num_training_samples)

        features = torch.empty((num_samples, self.num_features))
        labels = torch.empty((num_samples,), dtype=torch.int)
        device = next(self.model.parameters()).device

        with torch.no_grad():
            current_num_samples = 0
            for input, label in train_loader:
                if current_num_samples >= num_samples:
                    break  # Ensure we don't process beyond the desired number of samples

                # Calculate how many samples we can process in this batch
                actual_batch_size = min(
                    input.shape[0], num_samples - current_num_samples
                )
                input = input[:actual_batch_size]
                label = label[:actual_batch_size]

                input = input.to(device)

                feature = self.model.forward_head(
                    self.model.forward_features(input), pre_logits=True
                )

                end = current_num_samples + actual_batch_size

                features[current_num_samples:end] = feature.detach().cpu()
                labels[current_num_samples:end] = label.detach().cpu()

                current_num_samples += actual_batch_size

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
            print("Trying", jitter_eps, "...")
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1]
                ).unsqueeze(0)

                jittered_classwise_cov_features = classwise_cov_features + jitter

                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features,
                    covariance_matrix=jittered_classwise_cov_features,
                )

                self.gmm = gmm
                self.classwise_probs = classwise_probs.cuda()
                self.gmm_loc = classwise_mean_features
                self.gmm_covariance_matrix = jittered_classwise_cov_features
            except RuntimeError:
                continue
            except ValueError:
                continue
            break

        print("Used jitter:", jitter_eps)

        assert self.gmm is not None
