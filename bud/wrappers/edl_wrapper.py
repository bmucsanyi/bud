from torch import nn

from bud.wrappers.model_wrapper import DirichletWrapper
import torch.nn.functional as F


class EDLWrapper(DirichletWrapper):
    """
    This module takes a model as input and creates an EDL model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        activation: str,
    ):
        super().__init__(model)

        if activation == "exp":
            self._activation = lambda x: x.clamp(-10, 10).exp()
        elif activation == "softplus":
            self._activation = F.softplus
        else:
            raise ValueError(f'Invalid activation "{activation}" provided')

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)
        alphas = self._activation(logits).add(1)  # [B, C]

        if self.training:
            return alphas

        return {
            "alpha": alphas,  # [B, C]
            "feature": features,  # [B, D]
        }
