import torch
from torch import nn

from bud.wrappers.model_wrapper import SpecialWrapper
from bud.utils import entropy


class EDLWrapper(SpecialWrapper):
    """
    This module takes a model as input and creates an EDL model from it.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__(model)

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)

        if self.training:
            return logits
        else:
            num_classes = logits.shape[1]

            alphas = logits.clamp(-10, 10).exp().add(1)  # [B, C]
            sum_alphas = alphas.sum(dim=1)  # [B]

            dirichlet_scaled_inverse_precision = num_classes / sum_alphas  # [B]

            mean_alphas = alphas.div(sum_alphas.unsqueeze(1))  # [B, C]
            digamma_term = torch.digamma(alphas + 1) - torch.digamma(
                sum_alphas + 1
            ).unsqueeze(
                1
            )  # [B, C]
            dirichlet_expected_entropy = -mean_alphas.mul(digamma_term).sum(
                dim=1
            )  # [B]

            dirichlet_entropy_of_expectation = entropy(mean_alphas)  # [B]

            dirichlet_mutual_information = (
                dirichlet_entropy_of_expectation - dirichlet_expected_entropy
            )

            return {
                "logit": logits,  # [B, C]
                "feature": features,  # [B, D]
                "dirichlet_scaled_inverse_precision": dirichlet_scaled_inverse_precision,  # [B]
                "dirichlet_expected_entropy": dirichlet_expected_entropy,  # [B]
                "dirichlet_entropy_of_expectation": dirichlet_entropy_of_expectation,  # [B]
                "dirichlet_mutual_information": dirichlet_mutual_information,  # [B]
            }
