from torch import Tensor, nn
import torch


class DUQLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.loss = nn.BCELoss()

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return self.loss(prediction, target)
