import torch
from torch import nn


class NonIsotropicVMFLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, log_likelihoods, targets):
        # Average over samples. The avg log likelihood is equal to the ELK(vmf, nivMF)
        log_num_samples = torch.log(
            torch.tensor(log_likelihoods.shape[0], device=log_likelihoods.device)
        )
        sim_batch_vs_class = (
            log_likelihoods.logsumexp(dim=0) - log_num_samples
        )  # [B, C]

        # Compute loss
        loss = (
            -sim_batch_vs_class.gather(dim=1, index=targets.unsqueeze(dim=1))
            + sim_batch_vs_class.logsumexp(dim=1)
        ).mean()

        return loss
