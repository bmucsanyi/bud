import torch
from torch import Tensor, nn


class CorrectnessPredictionLoss(nn.Module):
    def __init__(
        self,
        lambda_uncertainty_loss,
        is_ignore_task_loss,
        is_top5,
    ):
        super().__init__()
        self.task_loss = nn.CrossEntropyLoss(reduction="none")
        self.uncertainty_loss = nn.BCEWithLogitsLoss()
        self.lambda_uncertainty_loss = lambda_uncertainty_loss
        self.task_loss_multiplier = 1 - is_ignore_task_loss
        self.is_top5 = is_top5

    def forward(
        self,
        prediction_tuple: tuple,
        target: Tensor,
    ) -> Tensor:
        prediction, correctness_prediction = prediction_tuple

        task_loss_per_sample = self.task_loss(prediction, target)

        if self.is_top5:
            _, prediction_argmax_top5 = torch.topk(prediction, 5, dim=1)
            expanded_gt_hard_labels = target.unsqueeze(dim=1).expand_as(
                prediction_argmax_top5
            )
            correctness = (
                prediction_argmax_top5.eq(expanded_gt_hard_labels).max(dim=1)[0].float()
            )
        else:
            correctness = prediction.argmax(dim=-1).eq(target).float()

        uncertainty_loss = self.uncertainty_loss(correctness_prediction, correctness)
        task_loss = task_loss_per_sample.mean()

        return (
            self.task_loss_multiplier * task_loss
            + self.lambda_uncertainty_loss * uncertainty_loss
        )
