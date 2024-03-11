"""Non-Isotropic vMF implementation as a wrapper class based on
https://github.com/mkirchhof/url"""

import torch
import torch.nn.functional as F
from torch import nn

from bud.utils import VonMisesFisher, vmf_log_norm_const
from bud.utils.model import NonNegativeRegressor
from bud.wrappers.model_wrapper import SpecialWrapper


class NonIsotropicvMFHead(nn.Module):
    def __init__(
        self,
        num_classes,
        num_features,
        num_hidden_features,
        num_mc_samples,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.num_hidden_features = num_hidden_features
        self.num_mc_samples = num_mc_samples
        self.register_buffer(
            "batch_kappa_scaler", torch.tensor(1.0, dtype=torch.float32)
        )

        # Initialize proxies
        self.unnormalized_class_mus = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes,
            bias=False,
        )

        self.unclipped_class_kappas = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes,
            bias=False,
        )

        self.kappa_predictor = NonNegativeRegressor(
            in_channels=self.num_features,
            width=self.num_hidden_features,
            depth=3,
            eps=1e-6,
        )

    def update_kappa_scaler(self, kappa_scaler):
        self.batch_kappa_scaler = torch.tensor(kappa_scaler, dtype=torch.float32)

    def forward(self, features):
        # Build vMFs from the features
        batch_mus = F.normalize(features, dim=-1)  # [B, L]

        batch_kappas = self.batch_kappa_scaler * self.kappa_predictor(
            features
        )  # [B, 1]

        # Get vMFs of classes
        class_mus = F.normalize(self.unnormalized_class_mus.weight, dim=-1)  # [C, L]
        class_kappas = torch.clamp(
            self.unclipped_class_kappas.weight, min=0.1
        )  # [C, L]

        # Draw samples
        vmf_distr = VonMisesFisher(loc=batch_mus, scale=batch_kappas)
        samples = vmf_distr.rsample(self.num_mc_samples)  # [S, B, L]

        # Calculate the term inside the exp() of the nivMF
        stretched_class_mus = class_mus * class_kappas  # [C, L]
        stretched_class_mu_norms = torch.norm(stretched_class_mus, dim=-1)  # [C]
        normalized_stretched_class_mus = F.normalize(
            stretched_class_mus, dim=-1
        )  # [C, L]

        # We want samples be an [S, B, C, L] tensor,
        # so that when we multiply it with the normalized_stretched_class_mus [C, L] tensor,
        # we do that across all combinations resulting in an [S, B, C] tensor
        stretched_samples = samples.unsqueeze(dim=2) * class_kappas  # [S, B, C, L]
        normalized_stretched_samples = F.normalize(
            stretched_samples, dim=-1
        )  # [S, B, C, L]

        cos_sims = torch.einsum(
            "...i,...i->...",
            normalized_stretched_samples,
            normalized_stretched_class_mus,
        )  # [S, B, C]

        # Calculate the remaining terms of the nivmf log likelihood
        log_norm_const = vmf_log_norm_const(stretched_class_mu_norms)
        log_det_term = class_kappas.log().sum(dim=-1) - stretched_class_mu_norms.log()
        log_likelihoods = (
            log_norm_const + log_det_term + stretched_class_mu_norms * cos_sims
        )  # [S, B, C]

        if self.training:
            return log_likelihoods  # negative ELK
        else:
            return {
                "logit": log_likelihoods.permute(1, 0, 2),
                "feature": features,
                "nivmf_inverse_kappa": 1 / batch_kappas.squeeze(),
            }


class NonIsotropicvMFWrapper(SpecialWrapper):
    """
    This module takes a model as input and creates a nivMF model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        num_hidden_features: int,
        initial_average_kappa,
    ):
        super().__init__(model)

        self.num_mc_samples = num_mc_samples
        self.num_hidden_features = num_hidden_features
        self.initial_average_kappa = initial_average_kappa

        self.classifier = NonIsotropicvMFHead(
            num_classes=self.num_classes,
            num_features=self.num_features,
            num_hidden_features=self.num_hidden_features,
            num_mc_samples=self.num_mc_samples,
        )

    def initialize_average_kappa(
        self, train_loader, amp_autocast, device, num_batches=10
    ):
        # Find out what kappa the model currently predicts on average
        prev_state = self.training
        self.eval()

        average_kappa = 0
        data_iter = iter(train_loader)

        with torch.no_grad():
            for _ in range(num_batches):
                input, _ = next(data_iter)

                if device is not None:
                    input = input.to(device)

                with amp_autocast():
                    inference_dict = self(input)
                kappa = 1 / inference_dict["nivmf_inverse_kappa"]
                average_kappa += kappa.mean().detach().cpu().item() / num_batches

        self.classifier.update_kappa_scaler(self.initial_average_kappa / average_kappa)
        self.train(prev_state)

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(
        self,
        num_hidden_features: int = None,
        num_mc_samples: int = None,
        *args,
        **kwargs,
    ):
        if num_mc_samples is not None:
            self.num_mc_samples = num_mc_samples

        if num_hidden_features is not None:
            self.num_hidden_features = num_hidden_features

        self.model.reset_classifier(*args, **kwargs)
        self.classifier = NonIsotropicvMFHead(
            num_classes=self.num_classes,
            num_features=self.num_features,
            num_hidden_features=self.num_hidden_features,
            num_mc_samples=self.num_mc_samples,
        )

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)

        return out
