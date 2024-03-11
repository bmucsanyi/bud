""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
                           and 2024 Bálint Mucsányi
"""

import faiss
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch import Tensor


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_pred_correct(output, target):
    """Computes whether each target label is the top-1 prediction of the output"""
    _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
    pred = pred.flatten()
    target = target.flatten()
    correct = pred == target
    return correct.float()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100 / batch_size
        for k in topk
    ]


def recall_at_one(features, targets, mode="matmul"):
    if mode == "matmul":
        # Expects tensors as inputs
        features = F.normalize(features, dim=-1)
        closest_idxes = features.matmul(features.transpose(-2, -1)).topk(2)[1][:, 1]
        closest_classes = targets[closest_idxes]
        is_same_class = (closest_classes == targets).int()
    elif mode == "faiss":
        # For big data, use faiss. Expects numpy arrays with float32 as inputs
        features = features.numpy()
        targets = targets.numpy()
        features = normalize(features, axis=1)
        faiss_search_index = faiss.IndexFlatIP(features.shape[-1])
        faiss_search_index.add(features)
        # Use 2, because the closest one will be the point itself
        _, closest_idxes = faiss_search_index.search(features, 2)
        closest_idxes = closest_idxes[:, 1]
        closest_classes = targets[closest_idxes]
        is_same_class = (closest_classes == targets).astype("int")
        is_same_class = torch.from_numpy(is_same_class)
    else:
        raise NotImplementedError(f"mode {mode} not implemented.")

    return is_same_class


def pct_cropped_has_bigger_pu(pu_orig, pu_cropped):
    return (pu_orig < pu_cropped).float().mean()


def entropy(probs, dim=-1):
    log_probs = probs.log()
    min_real = torch.finfo(log_probs.dtype).min
    log_probs = torch.clamp(log_probs, min=min_real)
    p_log_p = log_probs * probs

    return -p_log_p.sum(dim=dim)


def cross_entropy(probs_p, log_probs_q, dim=-1):
    p_log_q = probs_p * log_probs_q

    return -p_log_q.sum(dim=dim)


def kl_divergence(log_probs_p, log_probs_q, dim=-1):
    return (log_probs_p.exp() * (log_probs_p - log_probs_q)).sum(dim=dim)


def binary_log_probability(confidences, targets):
    confidences = confidences.clamp(min=1e-7, max=1 - 1e-7)
    return (
        targets * confidences.log() + (1 - targets) * (1 - confidences).log()
    ).mean()


def binary_brier(confidences, targets):
    return (-confidences.square() - targets + 2 * confidences * targets).mean()


def multiclass_log_probability(log_preds, targets):
    return -F.cross_entropy(log_preds, targets)


def multiclass_brier(log_preds, targets, is_soft_targets):
    preds = log_preds.exp()

    if not is_soft_targets:
        targets = F.one_hot(targets, num_classes=preds.shape[-1])

    return -(
        targets * (1 - 2 * preds + preds.square().sum(dim=-1, keepdim=True))
    ).mean()


def calculate_bin_metrics(
    confidences: Tensor, correctnesses: Tensor, num_bins: int = 10
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculates the binwise accuracies, confidences and proportions of samples.

    Args:
        confidences: Float tensor of shape (n,) containing predicted confidences.
        correctnesses: Float tensor of shape (n,) containing the true correctness labels.
        num_bins: Number of equally sized bins.

    Returns:
        bin_proportions: Float tensor of shape (num_bins,) containing proportion
            of samples in each bin. Sums up to 1.
        bin_confidences: Float tensor of shape (num_bins,) containing the average
            confidence for each bin.
        bin_accuracies: Float tensor of shape (num_bins,) containing the average
            accuracy for each bin.

    """
    correctnesses = correctnesses.float()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    indices = torch.bucketize(confidences.contiguous(), bin_boundaries) - 1
    indices = torch.clamp(indices, min=0, max=num_bins - 1)

    bin_counts = torch.zeros(num_bins, dtype=confidences.dtype)
    bin_counts.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))
    bin_proportions = bin_counts / bin_counts.sum()
    pos_counts = bin_counts > 0

    bin_confidences = torch.zeros(num_bins, dtype=confidences.dtype)
    bin_confidences.scatter_add_(dim=0, index=indices, src=confidences)
    bin_confidences[pos_counts] /= bin_counts[pos_counts]

    bin_accuracies = torch.zeros(num_bins, dtype=correctnesses.dtype)
    bin_accuracies.scatter_add_(dim=0, index=indices, src=correctnesses)
    bin_accuracies[pos_counts] /= bin_counts[pos_counts]

    return bin_proportions, bin_confidences, bin_accuracies


def calibration_error(
    confidences: Tensor, correctnesses: Tensor, num_bins: int, norm: str
) -> Tensor:
    """Computes the expected/maximum calibration error.

    Args:
        confidences: Float tensor of shape (n,) containing predicted confidences.
        correctnesses: Float tensor of shape (n,) containing the true correctness labels.
        num_bins: Number of equally sized bins.
        norm: Whether to return ECE (L1 norm) or MCE (inf norm)

    Returns:
        The ECE/MCE.

    """
    bin_proportions, bin_confidences, bin_accuracies = calculate_bin_metrics(
        confidences, correctnesses, num_bins
    )

    abs_diffs = (bin_accuracies - bin_confidences).abs()

    if norm == "l1":
        score = (bin_proportions * abs_diffs).sum()
    elif norm == "inf":
        score = abs_diffs.max()
    else:
        raise ValueError(f"Provided norm {norm} not l1 nor inf")

    return score


def centered_cov(x):
    n = x.shape[0]

    return 1 / (n - 1) * x.T @ x
