"""Shallow ensemble implementation as a wrapper class"""

import torch
from torch import Tensor, nn

from bud.wrappers.model_wrapper import PosteriorWrapper


class ShallowEnsembleClassifier(nn.Module):
    """Simple shallow ensemble classifier."""

    def __init__(self, num_heads, num_features, num_classes) -> None:
        super().__init__()
        self.shallow_classifiers = nn.Linear(num_features, num_classes * num_heads)
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        # Shape: [B, S, C]
        logits = self.shallow_classifiers(x).reshape(
            -1, self.num_heads, self.num_classes
        )

        return logits


class ShallowEnsembleWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates a shallow ensemble from it.
    """

    def __init__(
        self,
        model: nn.Module,
        num_heads: int,
    ):
        super().__init__(model)

        self.num_heads = num_heads
        self.classifier = ShallowEnsembleClassifier(
            num_heads=self.num_heads,
            num_features=self.num_features,
            num_classes=self.num_classes,
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_heads: int = None, *args, **kwargs):
        if num_heads is not None:
            self.num_heads = num_heads

        # Resets global pooling in `self.classifier`
        self.model.reset_classifier(*args, **kwargs)
        self.classifier = ShallowEnsembleClassifier(
            num_heads=self.num_heads,
            num_features=self.num_features,
            num_classes=self.num_classes,
        )
