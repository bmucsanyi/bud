import torch.nn.functional as F
from torch import nn


class BMACrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ce_loss = nn.NLLLoss()
        self.eps = 1e-10

    def forward(self, logits, targets):
        log_probs = (
            F.softmax(logits, dim=-1).mean(dim=1).add(self.eps).log()
        )  # [B, C]

        return self.ce_loss(log_probs, targets)
