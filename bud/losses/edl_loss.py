import torch
from torch import nn
import torch.nn.functional as F


class EDLLoss(nn.Module):
    def __init__(self, num_batches: int, num_classes: int) -> None:
        super().__init__()

        self.curr_step = 0
        self.max_step = 10 * num_batches
        self.register_buffer("uniform_alphas", torch.ones((num_classes,)))  # [C]
        self.register_buffer(
            "sum_uniform_alphas", torch.tensor(num_classes, dtype=torch.float32)
        )  # []
        self.register_buffer(
            "log_b_uniform_alphas",
            torch.lgamma(self.uniform_alphas).sum()
            - torch.lgamma(self.sum_uniform_alphas),
        )  # []

    def kullback_leibler_term(self, alpha_tildes: torch.Tensor) -> torch.Tensor:
        sum_alpha_tildes = alpha_tildes.sum(dim=1)  # [B]
        log_b_alpha_tildes = torch.lgamma(alpha_tildes).sum(dim=1) - torch.lgamma(
            sum_alpha_tildes
        )  # [B]

        digamma_sum_alpha_tildes = torch.digamma(sum_alpha_tildes)  # [B]
        digamma_alpha_tildes = torch.digamma(alpha_tildes)  # [B, C]

        kullback_leibler_term = (
            alpha_tildes.sub(self.uniform_alphas)  # [B, C]
            .mul(
                digamma_alpha_tildes.sub(digamma_sum_alpha_tildes.unsqueeze(1))
            )  # [B, C]
            .sum(dim=1)  # [B, 1]
            .sub(log_b_alpha_tildes)  # [B]
            .add(self.log_b_uniform_alphas)  # [B]
        )

        return kullback_leibler_term  # [B]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alphas = logits.clamp(-10, 10).exp().add(1)  # [B, C]
        sum_alphas = alphas.sum(dim=1)  # [B]
        mean_alphas = alphas.div(sum_alphas.unsqueeze(1))  # [B, C]

        targets_one_hot = F.one_hot(targets, num_classes=alphas.shape[1])  # [B, C]

        error_term = mean_alphas.sub(targets_one_hot).square().sum(dim=1)  # [B]
        variance_term = (
            mean_alphas.mul(1 - mean_alphas).sum(dim=1).div(sum_alphas + 1)
        )  # [B]

        annealing_coefficient = min(1, self.curr_step / self.max_step)
        self.curr_step += 1

        alpha_tildes = alphas.sub(1).mul(1 - targets_one_hot).add(1)  # [B, C]

        kullback_leibler_term = self.kullback_leibler_term(alpha_tildes)  # [B]

        return (
            error_term + variance_term + annealing_coefficient * kullback_leibler_term
        ).mean()  # []
