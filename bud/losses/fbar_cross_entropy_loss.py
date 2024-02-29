import torch
import torch.nn.functional as F
from torch import nn


class FBarCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        logits = F.log_softmax(logits, dim=-1).mean(dim=1)  # [B, C]

        return self.ce_loss(logits, targets)
