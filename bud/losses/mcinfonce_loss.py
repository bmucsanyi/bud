import torch
import torch.nn.functional as F
from torch import nn

from bud.utils import VonMisesFisher


class MCInfoNCELoss(nn.Module):
    def __init__(
        self,
        kappa_pos,
        num_mc_samples,
    ):
        super().__init__()

        self.kappa_pos = kappa_pos
        self.num_mc_samples = num_mc_samples
        self.log_num_mc_samples = torch.tensor(self.num_mc_samples).log()

    def forward(self, prediction_tuple, targets):
        if targets.shape[0] % 2 != 0:
            raise ValueError("Batch size must be even.")

        features, batch_kappas = prediction_tuple

        batch_mus = F.normalize(features, dim=-1)  # [B, D]

        samples = VonMisesFisher(batch_mus, batch_kappas).rsample(
            self.num_mc_samples
        )  # [S, B, D]

        # Calculate similarities
        similarities = self.kappa_pos * samples @ samples.transpose(-2, -1)  # [S, B, B]

        # Build positive and negative masks
        targets = torch.arange(
            targets.shape[0] // 2, device=targets.device, dtype=torch.int64
        ).repeat_interleave(
            2
        )  # [B]
        mask = (targets.unsqueeze(dim=1) == targets.unsqueeze(dim=0)).float()  # [B, B]
        positive_mask = mask - torch.eye(mask.shape[0], device=targets.device)
        other_mask = 1 - torch.eye(mask.shape[0], device=targets.device)

        positive_mask_complement = (~positive_mask.bool()).float()
        other_mask_complement = torch.eye(mask.shape[0], device=targets.device)

        # Terms with mask = 0 should be ignored in the sum
        positive_mask_add = positive_mask_complement.mul(-torch.inf).nan_to_num(0)
        other_mask_add = other_mask_complement.mul(-torch.inf).nan_to_num(0)

        # Calculate the standard log contrastive loss for each vMF sample
        log_infonce_per_sample_per_example = (
            similarities * positive_mask + positive_mask_add
        ).logsumexp(dim=-1) - (
            (similarities * other_mask + other_mask_add).logsumexp(dim=-1)
            - torch.tensor(targets.shape[0], device=targets.device).log()
        )  # [S, B]
        ## [K, 1]

        # Average over the samples (we actually want a logmeanexp, that's why we subtract
        # log(n_samples))
        log_infonce_per_example = (
            log_infonce_per_sample_per_example.logsumexp(0) - self.log_num_mc_samples
        )  # [B]

        # Calculate loss
        log_infonce = log_infonce_per_example.mean()  # []
        return -log_infonce
