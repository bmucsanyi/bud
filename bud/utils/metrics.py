"""Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
                           and 2024 Bálint Mucsányi
"""

import faiss
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


def area_under_lift_curve(
    uncertainties: Tensor, correctnesses: Tensor, reverse_sort: bool = False
) -> Tensor:
    correctnesses = correctnesses.float()
    batch_size = correctnesses.shape[0]

    if reverse_sort:
        sorted_idx = torch.argsort(
            uncertainties, descending=True
        )  # Most uncertain indices first
    else:
        sorted_idx = torch.argsort(uncertainties)  # Most certain indices first

    sorted_correctnesses = correctnesses[sorted_idx]
    lift = torch.zeros((batch_size,), dtype=torch.float32)
    accuracy = correctnesses.mean()
    lift[0] = sorted_correctnesses[0] / accuracy

    for i in range(1, batch_size):
        lift[i] = (i * lift[i - 1] + sorted_correctnesses[i] / accuracy) / (i + 1)

    step = 1 / batch_size

    return lift.sum() * step - 1


def relative_area_under_lift_curve(
    uncertainties: Tensor, correctnesses: Tensor
) -> Tensor:
    area = area_under_lift_curve(uncertainties, correctnesses)
    area_opt = area_under_lift_curve(correctnesses, correctnesses, reverse_sort=True)

    return area / area_opt


def dempster_shafer_metric(logits: Tensor) -> Tensor:
    num_classes = logits.shape[-1]
    belief_mass = logits.exp().sum(dim=-1)  # [B]
    dempster_shafer_value = num_classes / (belief_mass + num_classes)

    return dempster_shafer_value


def centered_cov(x):
    n = x.shape[0]

    return 1 / (n - 1) * x.T @ x


# https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance/blob/main/utils/uncertainty_metrics.py


def area_under_risk_coverage_curve(
    uncertainties: Tensor, correctnesses: Tensor
) -> Tensor:
    sorted_indices = torch.argsort(uncertainties)
    correctnesses = correctnesses[sorted_indices]
    total_samples = uncertainties.shape[0]
    aurc = torch.tensor(0.0)
    incorrect_num = 0

    for i in range(total_samples):
        incorrect_num += 1 - correctnesses[i]
        aurc += incorrect_num / (i + 1)

    aurc = aurc / total_samples

    return aurc


def excess_area_under_risk_coverage_curve(
    uncertainties: Tensor, correctnesses: Tensor
) -> Tensor:
    aurc = area_under_risk_coverage_curve(uncertainties, correctnesses)

    accuracy = correctnesses.float().mean()
    risk = 1 - accuracy
    # From https://arxiv.org/abs/1805.08206 :
    optimal_aurc = risk + (1 - risk) * torch.log(1 - risk)

    return aurc - optimal_aurc


def coverage_for_accuracy(
    uncertainties: Tensor,
    correctnesses: Tensor,
    accuracy: float = 0.95,
    start_index: int = 200,
) -> Tensor:
    sorted_indices = torch.argsort(uncertainties)
    correctnesses = correctnesses[sorted_indices]

    cumsum_correctnesses = torch.cumsum(correctnesses, dim=0)
    num_samples = cumsum_correctnesses.shape[0]
    cummean_correctnesses = cumsum_correctnesses / torch.arange(1, num_samples + 1)
    coverage_for_accuracy = torch.argmax((cummean_correctnesses < accuracy).float())

    # To ignore statistical noise, start measuring at an index greater than 0
    coverage_for_accuracy_nonstrict = (
        torch.argmax((cummean_correctnesses[start_index:] < accuracy).float())
        + start_index
    )
    if coverage_for_accuracy_nonstrict > start_index:
        # If they were the same, even the first non-noisy measurement didn't satisfy the risk, so its coverage is undue,
        # use the original index. Otherwise, use the non-strict to diffuse noisiness.
        coverage_for_accuracy = coverage_for_accuracy_nonstrict

    coverage_for_accuracy = coverage_for_accuracy / num_samples
    return coverage_for_accuracy
