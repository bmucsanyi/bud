""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.1, reduction="mean"):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert smoothing < 1.0
        assert reduction in ("mean", "sum", "none")
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self, reduction):
        super(SoftTargetCrossEntropyLoss, self).__init__()

        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
