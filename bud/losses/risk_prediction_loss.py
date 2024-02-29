from torch import Tensor, nn


class RiskPredictionLoss(nn.Module):
    def __init__(
        self,
        lambda_uncertainty_loss,
        is_detach,
        is_ignore_task_loss,
    ):
        super().__init__()

        self.task_loss = nn.CrossEntropyLoss(reduction="none")
        self.uncertainty_loss = nn.MSELoss()
        self.lambda_uncertainty_loss = lambda_uncertainty_loss
        self.is_detach = is_detach
        self.task_loss_multiplier = 1 - is_ignore_task_loss

    def forward(
        self,
        prediction_tuple: tuple,
        target: Tensor,
    ) -> Tensor:
        prediction, risk_prediction = prediction_tuple

        task_loss_per_sample = self.task_loss(prediction, target)

        if self.is_detach:
            task_loss_target = task_loss_per_sample.detach()
        else:
            task_loss_target = task_loss_per_sample

        uncertainty_loss = self.uncertainty_loss(risk_prediction, task_loss_target)
        task_loss = task_loss_per_sample.mean()

        return (
            self.task_loss_multiplier * task_loss
            + self.lambda_uncertainty_loss * uncertainty_loss
        )
