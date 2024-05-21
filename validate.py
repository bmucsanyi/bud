#!/usr/bin/env python3
"""ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for
evaluating pretrained models or training checkpoints against ImageNet or similarly
organized image datasets. It prioritizes canonical PyTorch, standard Python style, and
good performance. Repurpose as you see fit.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
                           and 2024 Bálint Mucsányi (https://github.com/bmucsanyi)
"""

import os
import time
import warnings
from numbers import Number

import torch
import torch.nn.functional as F
import torch.nn.parallel
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics import auc
from torchmetrics.functional.classification import binary_auroc as auroc

from bud.utils import (
    AverageMeter,
    binary_brier,
    binary_log_probability,
    calibration_error,
    entropy,
    kl_divergence,
    multiclass_brier,
    multiclass_log_probability,
    recall_at_one,
    dempster_shafer_metric,
    relative_area_under_lift_curve,
)
from bud.wrappers import (
    BaseCorrectnessPredictionWrapper,
    DDUWrapper,
    BaseLossPredictionWrapper,
    DeepEnsembleWrapper,
    DUQWrapper,
    MahalanobisWrapper,
    MCInfoNCEWrapper,
    NonIsotropicvMFWrapper,
    DirichletWrapper,
)

# Ignore constant input warning for correlation coefficients.
# This is expected to happen for all models without multiple
# sampled logits.
warnings.filterwarnings("ignore", category=ConstantInputWarning)

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

has_compile = hasattr(torch, "compile")


def evaluate_bulk(
    model,
    loaders,
    device,
    amp_autocast,
    key_prefix,
    temp_folder,
    is_same_task,
    is_upstream,
    args,
):
    metrics = {}

    for name, loader in loaders.items():
        metrics[name] = evaluate(
            model=model,
            loader=loader,
            device=device,
            amp_autocast=amp_autocast,
            key_prefix="",
            temp_folder=temp_folder,
            is_same_task=is_same_task,
            is_upstream=is_upstream,
            args=args,
        )

    # Summarize results
    flattened_metrics = add_average_and_flatten(results=metrics, key_prefix=key_prefix)

    return flattened_metrics


def add_average_and_flatten(results, key_prefix):
    # Summarize results
    avg_results = {}
    first_loader_results = results[list(results.keys())[0]]
    for key in first_loader_results:
        if isinstance(first_loader_results[key], Number):
            avg_results[key] = (
                torch.tensor(
                    [loader_result[key] for _, loader_result in results.items()]
                )
                .mean()
                .item()
            )
    results["avg"] = avg_results

    # Flatten output
    flattened_results = {}
    for name, dict in results.items():
        for key, value in dict.items():
            flattened_results[f"{key_prefix}_{name}_{key}"] = value

    return flattened_results


def evaluate(
    model,
    loader,
    device,
    amp_autocast,
    key_prefix,
    temp_folder,
    is_same_task,
    is_upstream,
    args,
):
    model.eval()
    torch.set_grad_enabled(mode=False)

    assert not (
        is_upstream and not is_same_task
    ), "The upstream dataset has the same task by definition"

    label_shape = next(iter(loader))[1].shape
    is_soft_labels = len(label_shape) == 2

    estimates, log_probs, targets, times = get_bundle(
        model=model,
        loader=loader,
        device=device,
        amp_autocast=amp_autocast,
        is_soft_labels=is_soft_labels,
        is_same_task=is_same_task,
        args=args,
    )

    metrics = times

    metrics = evaluate_on_tasks(
        model=model,
        estimates=estimates,
        log_probs=log_probs,
        targets=targets,
        metrics=metrics,
        is_same_task=is_same_task,
        is_soft_labels=is_soft_labels,
        args=args,
    )

    if is_upstream:
        # Save ingredients to disk
        max_num_indices = len(targets["gt_zero_shot_correctnesses"])
        num_indices = min(max_num_indices, args.max_num_id_ood_eval_samples // 2)
        path_indices = f"bud/data/{num_indices}_indices_out_of_{max_num_indices}.pt"
        if os.path.exists(path_indices):
            indices = torch.load(path_indices)
        else:
            indices = torch.randperm(max_num_indices)[:num_indices]
            torch.save(indices, path_indices)

        upstream_dict = {
            "upstream_estimates": filter_entries(estimates, indices),
            "upstream_targets": filter_entries(targets, indices),
            "upstream_is_soft_labels": is_soft_labels,
        }

        if not isinstance(model, MCInfoNCEWrapper):
            upstream_dict["upstream_log_probs"] = filter_entries(log_probs, indices)

        torch.save(upstream_dict, f"{temp_folder}/upstream_dict.pt")
    else:
        # Load ingredients from disk
        upstream_dict = torch.load(f"{temp_folder}/upstream_dict.pt")
        upstream_estimates = upstream_dict["upstream_estimates"]

        if not isinstance(model, MCInfoNCEWrapper):
            upstream_log_probs = upstream_dict["upstream_log_probs"]

        upstream_targets = upstream_dict["upstream_targets"]
        upstream_is_soft_labels = upstream_dict["upstream_is_soft_labels"]

        # Make both upstream and downstream tensors the same size to get a 50/50 split
        num_upstream_indices = len(upstream_targets["gt_zero_shot_correctnesses"])
        max_num_downstream_indices = len(targets["gt_zero_shot_correctnesses"])
        num_indices_to_keep = min(num_upstream_indices, max_num_downstream_indices)

        # For upstream, we can just use [:num_samples_keep] in the following, because
        # it's already shuffled. For downstream, let's use random indices
        path_downstream_indices = (
            f"bud/data/{num_indices_to_keep}_indices_out_of"
            f"_{max_num_downstream_indices}.pt"
        )
        if os.path.exists(path_downstream_indices):
            downstream_indices = torch.load(path_downstream_indices)
        else:
            downstream_indices = torch.randperm(max_num_downstream_indices)[
                :num_indices_to_keep
            ]
            torch.save(downstream_indices, path_downstream_indices)

        upstream_estimates = truncate_entries(upstream_estimates, num_indices_to_keep)
        upstream_targets = truncate_entries(upstream_targets, num_indices_to_keep)

        if not isinstance(model, MCInfoNCEWrapper):
            upstream_log_probs = truncate_entries(
                upstream_log_probs, num_indices_to_keep
            )
            downstream_log_probs = filter_entries(log_probs, downstream_indices)

        downstream_estimates = filter_entries(estimates, downstream_indices)
        downstream_targets = filter_entries(targets, downstream_indices)

        # Mix ingredients (remember, we're cooking!)
        mixed_estimates = concatenate_values(upstream_estimates, downstream_estimates)

        if not isinstance(model, MCInfoNCEWrapper):
            mixed_log_probs = concatenate_values(
                upstream_log_probs, downstream_log_probs
            )
        else:
            mixed_log_probs = {}

        mixed_targets = concatenate_values(
            upstream_targets, downstream_targets, keys_to_exclude=["gt_soft_labels"]
        )

        # Update joint targets
        mixed_targets["gt_oodness"] = torch.cat(
            [torch.zeros((num_indices_to_keep,)), torch.ones((num_indices_to_keep,))]
        ).int()
        if is_same_task:
            if upstream_is_soft_labels and not is_soft_labels:
                num_classes = upstream_targets["gt_soft_labels"].shape[1]
                mixed_targets["gt_soft_labels"] = torch.cat(
                    [
                        upstream_targets["gt_soft_labels"],
                        F.one_hot(
                            downstream_targets["gt_hard_labels"],
                            num_classes=num_classes,
                        ),
                    ]
                )

                if not isinstance(model, MCInfoNCEWrapper):
                    mixed_targets["gt_soft_fbar_correctnesses"] = torch.cat(
                        [
                            upstream_targets["gt_soft_fbar_correctnesses"],
                            downstream_targets["gt_hard_fbar_correctnesses"],
                        ]
                    )
                    mixed_targets["gt_soft_fbar_correctnesses_top5"] = torch.cat(
                        [
                            upstream_targets["gt_soft_fbar_correctnesses_top5"],
                            downstream_targets["gt_hard_fbar_correctnesses_top5"],
                        ]
                    )

                    mixed_targets["gt_soft_bma_correctnesses"] = torch.cat(
                        [
                            upstream_targets["gt_soft_bma_correctnesses"],
                            downstream_targets["gt_hard_bma_correctnesses"],
                        ]
                    )
                    mixed_targets["gt_soft_bma_correctnesses_top5"] = torch.cat(
                        [
                            upstream_targets["gt_soft_bma_correctnesses_top5"],
                            downstream_targets["gt_hard_bma_correctnesses_top5"],
                        ]
                    )
            elif not upstream_is_soft_labels and is_soft_labels:
                num_classes = downstream_targets["gt_soft_labels"].shape[1]
                mixed_targets["gt_soft_labels"] = torch.cat(
                    [
                        F.one_hot(
                            upstream_targets["gt_hard_labels"],
                            num_classes=num_classes,
                        ),
                        downstream_targets["gt_soft_labels"],
                    ]
                )

                if not isinstance(model, MCInfoNCEWrapper):
                    mixed_targets["gt_soft_fbar_correctnesses"] = torch.cat(
                        [
                            upstream_targets["gt_hard_fbar_correctnesses"],
                            downstream_targets["gt_soft_fbar_correctnesses"],
                        ]
                    )
                    mixed_targets["gt_soft_fbar_correctnesses_top5"] = torch.cat(
                        [
                            upstream_targets["gt_hard_fbar_correctnesses_top5"],
                            downstream_targets["gt_soft_fbar_correctnesses_top5"],
                        ]
                    )

                    mixed_targets["gt_soft_bma_correctnesses"] = torch.cat(
                        [
                            upstream_targets["gt_hard_bma_correctnesses"],
                            downstream_targets["gt_soft_bma_correctnesses"],
                        ]
                    )
                    mixed_targets["gt_soft_bma_correctnesses_top5"] = torch.cat(
                        [
                            upstream_targets["gt_hard_bma_correctnesses_top5"],
                            downstream_targets["gt_soft_bma_correctnesses_top5"],
                        ]
                    )
            elif upstream_is_soft_labels and is_soft_labels:
                mixed_targets["gt_soft_labels"] = torch.cat(
                    [
                        upstream_targets["gt_soft_labels"],
                        downstream_targets["gt_soft_labels"],
                    ]
                )

        metrics = evaluate_on_tasks(
            model=model,
            estimates=mixed_estimates,
            log_probs=mixed_log_probs,
            targets=mixed_targets,
            metrics=metrics,
            is_same_task=is_same_task,
            is_soft_labels=is_soft_labels,
            args=args,
            upstream_is_soft_labels=upstream_is_soft_labels,
        )

    if key_prefix:
        for metric_name in list(metrics.keys()):
            metrics[f"{key_prefix}_{metric_name}"] = metrics.pop(metric_name)

    torch.set_grad_enabled(mode=True)

    return metrics


def filter_entries(estimates, indices):
    filtered_estimates = estimates.copy()

    for estimator_name, estimate in filtered_estimates.items():
        filtered_estimates[estimator_name] = estimate[indices]

    return filtered_estimates


def truncate_entries(estimates, num_indices_to_keep):
    truncated_estimates = estimates.copy()

    for estimator_name, estimate in truncated_estimates.items():
        truncated_estimates[estimator_name] = estimate[:num_indices_to_keep]

    return truncated_estimates


def concatenate_values(upstream_dict, downstream_dict, keys_to_exclude=None):
    if keys_to_exclude is None:
        keys_to_exclude = []

    common_keys = upstream_dict.keys() & downstream_dict.keys()
    result = {
        key: torch.cat([upstream_dict[key], downstream_dict[key]], dim=0)
        for key in common_keys
        if key not in keys_to_exclude
    }

    return result


def evaluate_on_tasks(
    model,
    estimates,
    log_probs,
    targets,
    metrics,
    is_same_task,
    is_soft_labels,
    args,
    upstream_is_soft_labels=None,
):
    is_mixed = upstream_is_soft_labels is not None

    metrics |= evaluate_on_correctness_of_prediction(
        model=model,
        estimates=estimates,
        targets=targets,
        is_same_task=is_same_task,
        is_soft_labels=is_soft_labels,
        args=args,
        upstream_is_soft_labels=upstream_is_soft_labels,
    )

    metrics |= evaluate_on_abstained_prediction(
        model=model,
        estimates=estimates,
        targets=targets,
        is_same_task=is_same_task,
        is_soft_labels=is_soft_labels,
        args=args,
        upstream_is_soft_labels=upstream_is_soft_labels,
    )

    if is_mixed:
        metrics |= evaluate_on_ood_detection(
            estimates=estimates,
            targets=targets,
            args=args,
        )

    if not isinstance(model, MCInfoNCEWrapper):
        metrics |= evaluate_on_proper_scoring_and_calibration(
            model=model,
            estimates=estimates,
            log_probs=log_probs,
            targets=targets,
            is_same_task=is_same_task,
            is_soft_labels=is_soft_labels,
            args=args,
            upstream_is_soft_labels=upstream_is_soft_labels,
        )

    metrics |= evaluate_on_bregman(
        model=model,
        estimates=estimates,
        targets=targets,
        is_same_task=is_same_task,
        is_soft_labels=is_soft_labels,
        args=args,
        upstream_is_soft_labels=upstream_is_soft_labels,
    )
    metrics |= evaluate_on_correlation_of_decompositions(
        model=model,
        estimates=estimates,
        targets=targets,
        is_same_task=is_same_task,
        is_soft_labels=is_soft_labels,
        args=args,
        upstream_is_soft_labels=upstream_is_soft_labels,
    )

    return metrics


def evaluate_on_correctness_of_prediction(
    model,
    estimates,
    targets,
    is_same_task,
    is_soft_labels,
    args,
    upstream_is_soft_labels,
):
    is_mixed = upstream_is_soft_labels is not None

    # For correctness of prediction, one of the datasets being soft is enough
    if is_mixed:
        is_soft_labels = is_soft_labels or upstream_is_soft_labels

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed else ""

    gt_zero_shot_correctnesses = targets["gt_zero_shot_correctnesses"]

    if is_same_task and not isinstance(model, MCInfoNCEWrapper):
        gt_hard_fbar_correctnesses = targets["gt_hard_fbar_correctnesses"]
        gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

        gt_hard_fbar_correctnesses_top5 = targets["gt_hard_fbar_correctnesses_top5"]
        gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

        if is_soft_labels:
            gt_soft_fbar_correctnesses = targets["gt_soft_fbar_correctnesses"]
            gt_soft_bma_correctnesses = targets["gt_soft_bma_correctnesses"]

            gt_soft_fbar_correctnesses_top5 = targets["gt_soft_fbar_correctnesses_top5"]
            gt_soft_bma_correctnesses_top5 = targets["gt_soft_bma_correctnesses_top5"]

    for estimator_name in estimates:
        # In `estimates`, we have *uncertainty* estimates: higher signals more uncertain.
        # For correctness of prediction, we need *certainty* estimates: the AUROC is high
        # if there exists a threshold for which all certain samples are correct (1)
        # and all others are incorrect (0).

        estimate = -estimates[estimator_name]
        metrics[f"{key_prefix}{estimator_name}_auroc_zero_shot_correctness"] = auroc(
            estimate, gt_zero_shot_correctnesses
        ).item()

        if is_same_task and not isinstance(model, MCInfoNCEWrapper):
            metrics[
                f"{key_prefix}{estimator_name}_auroc_hard_fbar_correctness"
            ] = calculate_auroc(
                estimate, gt_hard_fbar_correctnesses, args, soft=False
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness"
            ] = calculate_auroc(
                estimate, gt_hard_bma_correctnesses, args, soft=False
            ).item()

            metrics[
                f"{key_prefix}{estimator_name}_auroc_hard_fbar_correctness_top5"
            ] = calculate_auroc(
                estimate, gt_hard_fbar_correctnesses_top5, args, soft=False
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness_top5"
            ] = calculate_auroc(
                estimate, gt_hard_bma_correctnesses_top5, args, soft=False
            ).item()

            if is_soft_labels:
                metrics[
                    f"{key_prefix}{estimator_name}_auroc_soft_fbar_correctness"
                ] = calculate_auroc(
                    estimate, gt_soft_fbar_correctnesses, args, soft=True
                ).item()
                metrics[
                    f"{key_prefix}{estimator_name}_auroc_soft_bma_correctness"
                ] = calculate_auroc(
                    estimate, gt_soft_bma_correctnesses, args, soft=True
                ).item()

                metrics[
                    f"{key_prefix}{estimator_name}_auroc_soft_fbar_correctness_top5"
                ] = calculate_auroc(
                    estimate, gt_soft_fbar_correctnesses_top5, args, soft=True
                ).item()
                metrics[
                    f"{key_prefix}{estimator_name}_auroc_soft_bma_correctness_top5"
                ] = calculate_auroc(
                    estimate, gt_soft_bma_correctnesses_top5, args, soft=True
                ).item()

    # Performance metrics
    metrics["recall_at_1"] = gt_zero_shot_correctnesses.float().mean().item()

    if is_same_task and not isinstance(model, MCInfoNCEWrapper):
        metrics[f"{key_prefix}hard_fbar_accuracy"] = (
            targets["gt_hard_fbar_correctnesses"].float().mean().item()
        )
        metrics[f"{key_prefix}hard_bma_accuracy"] = (
            targets["gt_hard_bma_correctnesses"].float().mean().item()
        )

        metrics[f"{key_prefix}hard_fbar_accuracy_top5"] = (
            targets["gt_hard_fbar_correctnesses_top5"].float().mean().item()
        )
        metrics[f"{key_prefix}hard_bma_accuracy_top5"] = (
            targets["gt_hard_bma_correctnesses_top5"].float().mean().item()
        )

        if is_soft_labels:
            metrics[f"{key_prefix}soft_fbar_accuracy"] = (
                targets["gt_soft_fbar_correctnesses"].mean().item()
            )
            metrics[f"{key_prefix}soft_bma_accuracy"] = (
                targets["gt_soft_bma_correctnesses"].mean().item()
            )

            metrics[f"{key_prefix}soft_fbar_accuracy_top5"] = (
                targets["gt_soft_fbar_correctnesses_top5"].mean().item()
            )
            metrics[f"{key_prefix}soft_bma_accuracy_top5"] = (
                targets["gt_soft_bma_correctnesses_top5"].mean().item()
            )

            probs = targets["gt_soft_labels"]
            max_labels = probs.max(dim=1)[0]
            metrics[f"{key_prefix}best_soft_accuracy"] = max_labels.mean().item()
            # Calculate best soft AUROC score
            # Calculate best possible AUROC values
            # metrics[
            #     f"{key_prefix}best_soft_fbar_auroc"
            # ] = get_best_soft_auroc_scores(targets["gt_soft_fbar_correctnesses"], args)

            # if isinstance(model, PosteriorWrapper):
            #     metrics[
            #         f"{key_prefix}best_soft_bma_auroc"
            #     ] = get_best_soft_auroc_scores(
            #         targets["gt_soft_bma_correctnesses"], args
            #     )

    return metrics


def evaluate_on_abstained_prediction(
    model,
    estimates,
    targets,
    is_same_task,
    is_soft_labels,
    args,
    upstream_is_soft_labels,
):
    is_mixed = upstream_is_soft_labels is not None

    # For correctness of prediction, one of the datasets being soft is enough
    if is_mixed:
        is_soft_labels = is_soft_labels or upstream_is_soft_labels

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed else ""

    gt_zero_shot_correctnesses = targets["gt_zero_shot_correctnesses"]
    num_samples = gt_zero_shot_correctnesses.shape[0]

    if is_same_task and not isinstance(model, MCInfoNCEWrapper):
        gt_hard_fbar_correctnesses = targets["gt_hard_fbar_correctnesses"]
        gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

        gt_hard_fbar_correctnesses_top5 = targets["gt_hard_fbar_correctnesses_top5"]
        gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

        if is_soft_labels:
            gt_soft_fbar_correctnesses = targets["gt_soft_fbar_correctnesses"]
            gt_soft_bma_correctnesses = targets["gt_soft_bma_correctnesses"]

            gt_soft_fbar_correctnesses_top5 = targets["gt_soft_fbar_correctnesses_top5"]
            gt_soft_bma_correctnesses_top5 = targets["gt_soft_bma_correctnesses_top5"]

    increasing_count = torch.arange(num_samples) + 1
    increasing_fraction = increasing_count / num_samples

    for estimator_name in estimates:
        estimate = estimates[estimator_name]

        indices = estimate.argsort()  # from least uncertain to most uncertain

        metrics[
            f"{key_prefix}{estimator_name}_cumulative_zero_shot_abstinence_auc"
        ] = auc(
            increasing_fraction,
            gt_zero_shot_correctnesses[indices].cumsum(dim=0) / increasing_count,
        )
        metrics[
            f"{key_prefix}{estimator_name}_zero_shot_raulc"
        ] = relative_area_under_lift_curve(estimate, gt_zero_shot_correctnesses).item()

        if is_same_task and not isinstance(model, MCInfoNCEWrapper):
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_hard_fbar_abstinence_auc"
            ] = auc(
                increasing_fraction,
                gt_hard_fbar_correctnesses[indices].cumsum(dim=0) / increasing_count,
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_hard_bma_abstinence_auc"
            ] = auc(
                increasing_fraction,
                gt_hard_bma_correctnesses[indices].cumsum(dim=0) / increasing_count,
            )

            metrics[
                f"{key_prefix}{estimator_name}_cumulative_hard_fbar_abstinence_auc_top5"
            ] = auc(
                increasing_fraction,
                gt_hard_fbar_correctnesses_top5[indices].cumsum(dim=0)
                / increasing_count,
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_hard_bma_abstinence_auc_top5"
            ] = auc(
                increasing_fraction,
                gt_hard_bma_correctnesses_top5[indices].cumsum(dim=0)
                / increasing_count,
            )

            metrics[
                f"{key_prefix}{estimator_name}_hard_fbar_raulc"
            ] = relative_area_under_lift_curve(estimate, gt_hard_fbar_correctnesses)
            metrics[
                f"{key_prefix}{estimator_name}_hard_bma_raulc"
            ] = relative_area_under_lift_curve(
                estimate,
                gt_hard_bma_correctnesses,
            )

            metrics[
                f"{key_prefix}{estimator_name}_hard_fbar_raulc_top5"
            ] = relative_area_under_lift_curve(
                estimate,
                gt_hard_fbar_correctnesses_top5,
            )
            metrics[
                f"{key_prefix}{estimator_name}_hard_bma_raulc_top5"
            ] = relative_area_under_lift_curve(
                estimate,
                gt_hard_bma_correctnesses_top5,
            )

            if is_soft_labels:
                metrics[
                    f"{key_prefix}{estimator_name}_cumulative_soft_fbar_abstinence_auc"
                ] = auc(
                    increasing_fraction,
                    gt_soft_fbar_correctnesses[indices].cumsum(dim=0)
                    / increasing_count,
                )
                metrics[
                    f"{key_prefix}{estimator_name}_cumulative_soft_bma_abstinence_auc"
                ] = auc(
                    increasing_fraction,
                    gt_soft_bma_correctnesses[indices].cumsum(dim=0) / increasing_count,
                )

                metrics[
                    f"{key_prefix}{estimator_name}_cumulative_soft_fbar_abstinence_auc_top5"
                ] = auc(
                    increasing_fraction,
                    gt_soft_fbar_correctnesses_top5[indices].cumsum(dim=0)
                    / increasing_count,
                )
                metrics[
                    f"{key_prefix}{estimator_name}_cumulative_soft_bma_abstinence_auc_top5"
                ] = auc(
                    increasing_fraction,
                    gt_soft_bma_correctnesses_top5[indices].cumsum(dim=0)
                    / increasing_count,
                )

                metrics[
                    f"{key_prefix}{estimator_name}_soft_fbar_raulc"
                ] = relative_area_under_lift_curve(estimate, gt_soft_fbar_correctnesses)
                metrics[
                    f"{key_prefix}{estimator_name}_soft_bma_raulc"
                ] = relative_area_under_lift_curve(
                    estimate,
                    gt_soft_bma_correctnesses,
                )

                metrics[
                    f"{key_prefix}{estimator_name}_soft_fbar_raulc_top5"
                ] = relative_area_under_lift_curve(
                    estimate,
                    gt_soft_fbar_correctnesses_top5,
                )
                metrics[
                    f"{key_prefix}{estimator_name}_soft_bma_raulc_top5"
                ] = relative_area_under_lift_curve(
                    estimate,
                    gt_soft_bma_correctnesses_top5,
                )

    return metrics


def evaluate_on_ood_detection(estimates, targets, args):
    metrics = {}
    for estimator_name in estimates:
        metrics[f"mixed_{args.dataset_id}_{estimator_name}_auroc_oodness"] = auroc(
            estimates[estimator_name], targets["gt_oodness"]
        ).item()

    return metrics


def evaluate_on_proper_scoring_and_calibration(
    model,
    estimates,
    log_probs,
    targets,
    is_same_task,
    is_soft_labels,
    args,
    upstream_is_soft_labels,
):
    is_mixed = upstream_is_soft_labels is not None

    # For proper scoring and calibration, one of the datasets being soft is enough
    if is_mixed:
        is_soft_labels = is_soft_labels or upstream_is_soft_labels

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed else ""

    # Proper scoring and calibration for correctness of prediction
    correctness_estimator_names = [
        "one_minus_expected_max_probs",
        "one_minus_max_probs_of_fbar",
        "one_minus_max_probs_of_bma",
    ]

    if isinstance(model, DUQWrapper):
        correctness_estimator_names.append("duq_values")

    if isinstance(model, BaseCorrectnessPredictionWrapper):
        correctness_estimator_names.append("error_probabilities")

    gt_zero_shot_correctnesses = targets["gt_zero_shot_correctnesses"]

    if is_same_task:
        gt_hard_fbar_correctnesses = targets["gt_hard_fbar_correctnesses"]
        gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

        gt_hard_fbar_correctnesses_top5 = targets["gt_hard_fbar_correctnesses_top5"]
        gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

        if is_soft_labels:
            gt_soft_fbar_correctnesses = targets["gt_soft_fbar_correctnesses"]
            gt_soft_bma_correctnesses = targets["gt_soft_bma_correctnesses"]

            gt_soft_fbar_correctnesses_top5 = targets["gt_soft_fbar_correctnesses_top5"]
            gt_soft_bma_correctnesses_top5 = targets["gt_soft_bma_correctnesses_top5"]

    for estimator_name in correctness_estimator_names:
        estimate = estimates[estimator_name]

        estimate = 1 - estimate  # convert to correctness probability

        # Zero-shot correctness
        # Binary log probability scoring rule
        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_zero_shot_correctness"
        ] = binary_log_probability(estimate, gt_zero_shot_correctnesses).item()

        # Binary Brier scoring rule
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_zero_shot_correctness"
        ] = binary_brier(estimate, gt_zero_shot_correctnesses).item()

        # Binary ECE
        metrics[
            f"{key_prefix}{estimator_name}_ece_zero_shot_correctness"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_zero_shot_correctnesses,
            num_bins=15,
            norm="l1",
        ).item()

        # Binary MCE
        metrics[
            f"{key_prefix}{estimator_name}_mce_zero_shot_correctness"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_zero_shot_correctnesses,
            num_bins=15,
            norm="inf",
        ).item()

        # {Hard, Soft}-label correctness
        if is_same_task:
            metrics[
                f"{key_prefix}{estimator_name}_log_prob_score_hard_fbar_correctness"
            ] = binary_log_probability(estimate, gt_hard_fbar_correctnesses).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_hard_fbar_correctness"
            ] = binary_brier(estimate, gt_hard_fbar_correctnesses).item()
            metrics[
                f"{key_prefix}{estimator_name}_ece_hard_fbar_correctness"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses,
                num_bins=15,
                norm="l1",
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_mce_hard_fbar_correctness"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses,
                num_bins=15,
                norm="inf",
            ).item()

            metrics[
                f"{key_prefix}{estimator_name}_log_prob_score_hard_bma_correctness"
            ] = binary_log_probability(estimate, gt_hard_bma_correctnesses).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_hard_bma_correctness"
            ] = binary_brier(estimate, gt_hard_bma_correctnesses).item()
            metrics[
                f"{key_prefix}{estimator_name}_ece_hard_bma_correctness"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses,
                num_bins=15,
                norm="l1",
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_mce_hard_bma_correctness"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses,
                num_bins=15,
                norm="inf",
            ).item()

            metrics[
                f"{key_prefix}{estimator_name}_log_prob_score_hard_fbar_correctness_top5"
            ] = binary_log_probability(estimate, gt_hard_fbar_correctnesses_top5).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_hard_fbar_correctness_top5"
            ] = binary_brier(estimate, gt_hard_fbar_correctnesses_top5).item()
            metrics[
                f"{key_prefix}{estimator_name}_ece_hard_fbar_correctness_top5"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses_top5,
                num_bins=15,
                norm="l1",
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_mce_hard_fbar_correctness_top5"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses_top5,
                num_bins=15,
                norm="inf",
            ).item()

            metrics[
                f"{key_prefix}{estimator_name}_log_prob_score_hard_bma_correctness_top5"
            ] = binary_log_probability(estimate, gt_hard_bma_correctnesses_top5).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_hard_bma_correctness_top5"
            ] = binary_brier(estimate, gt_hard_bma_correctnesses_top5).item()
            metrics[
                f"{key_prefix}{estimator_name}_ece_hard_bma_correctness_top5"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses_top5,
                num_bins=15,
                norm="l1",
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_mce_hard_bma_correctness_top5"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses_top5,
                num_bins=15,
                norm="inf",
            ).item()

            if is_soft_labels:
                metrics[
                    f"{key_prefix}{estimator_name}_log_prob_score_soft_fbar_correctness"
                ] = binary_log_probability(estimate, gt_soft_fbar_correctnesses).item()
                metrics[
                    f"{key_prefix}{estimator_name}_brier_score_soft_fbar_correctness"
                ] = binary_brier(estimate, gt_soft_fbar_correctnesses).item()
                metrics[
                    f"{key_prefix}{estimator_name}_ece_soft_fbar_correctness"
                ] = calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_fbar_correctnesses,
                    num_bins=15,
                    norm="l1",
                ).item()
                metrics[
                    f"{key_prefix}{estimator_name}_mce_soft_fbar_correctness"
                ] = calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_fbar_correctnesses,
                    num_bins=15,
                    norm="inf",
                ).item()

                metrics[
                    f"{key_prefix}{estimator_name}_log_prob_score_soft_bma_correctness"
                ] = binary_log_probability(estimate, gt_soft_bma_correctnesses).item()
                metrics[
                    f"{key_prefix}{estimator_name}_brier_score_soft_bma_correctness"
                ] = binary_brier(estimate, gt_soft_bma_correctnesses).item()
                metrics[
                    f"{key_prefix}{estimator_name}_ece_soft_bma_correctness"
                ] = calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_bma_correctnesses,
                    num_bins=15,
                    norm="l1",
                ).item()
                metrics[
                    f"{key_prefix}{estimator_name}_mce_soft_bma_correctness"
                ] = calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_bma_correctnesses,
                    num_bins=15,
                    norm="inf",
                ).item()

                metrics[
                    f"{key_prefix}{estimator_name}_log_prob_score_soft_fbar_correctness_top5"
                ] = binary_log_probability(
                    estimate, gt_soft_fbar_correctnesses_top5
                ).item()
                metrics[
                    f"{key_prefix}{estimator_name}_brier_score_soft_fbar_correctness_top5"
                ] = binary_brier(estimate, gt_soft_fbar_correctnesses_top5).item()
                metrics[
                    f"{key_prefix}{estimator_name}_ece_soft_fbar_correctness_top5"
                ] = calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_fbar_correctnesses_top5,
                    num_bins=15,
                    norm="l1",
                ).item()
                metrics[
                    f"{key_prefix}{estimator_name}_mce_soft_fbar_correctness_top5"
                ] = calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_fbar_correctnesses_top5,
                    num_bins=15,
                    norm="inf",
                ).item()

                metrics[
                    f"{key_prefix}{estimator_name}_log_prob_score_soft_bma_correctness_top5"
                ] = binary_log_probability(
                    estimate, gt_soft_bma_correctnesses_top5
                ).item()
                metrics[
                    f"{key_prefix}{estimator_name}_brier_score_soft_bma_correctness_top5"
                ] = binary_brier(estimate, gt_soft_bma_correctnesses_top5).item()
                metrics[
                    f"{key_prefix}{estimator_name}_ece_soft_bma_correctness_top5"
                ] = calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_bma_correctnesses_top5,
                    num_bins=15,
                    norm="l1",
                ).item()
                metrics[
                    f"{key_prefix}{estimator_name}_mce_soft_bma_correctness_top5"
                ] = calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_bma_correctnesses_top5,
                    num_bins=15,
                    norm="inf",
                ).item()

    # Proper scoring for aleatoric uncertainty
    if is_same_task:
        gt_hard_labels = targets["gt_hard_labels"]

        metrics[
            f"{key_prefix}log_prob_score_hard_fbar_aleatoric"
        ] = multiclass_log_probability(log_probs["log_fbars"], gt_hard_labels).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_fbar_aleatoric"
        ] = multiclass_brier(
            log_probs["log_fbars"], gt_hard_labels, is_soft_targets=False
        ).item()

        metrics[
            f"{key_prefix}log_prob_score_hard_bma_aleatoric"
        ] = multiclass_log_probability(log_probs["log_bmas"], gt_hard_labels).item()
        metrics[f"{key_prefix}brier_score_hard_fbar_aleatoric"] = multiclass_brier(
            log_probs["log_bmas"], gt_hard_labels, is_soft_targets=False
        ).item()

        if is_soft_labels:
            gt_soft_labels = targets["gt_soft_labels"]

            metrics[
                f"{key_prefix}log_prob_score_soft_fbar_aleatoric"
            ] = multiclass_log_probability(
                log_probs["log_fbars"], gt_soft_labels
            ).item()
            metrics[f"{key_prefix}brier_score_soft_fbar_aleatoric"] = multiclass_brier(
                log_probs["log_fbars"], gt_soft_labels, is_soft_targets=True
            ).item()

            metrics[
                f"{key_prefix}log_prob_score_soft_bma_aleatoric"
            ] = multiclass_log_probability(log_probs["log_bmas"], gt_soft_labels).item()
            metrics[f"{key_prefix}brier_score_soft_fbar_aleatoric"] = multiclass_brier(
                log_probs["log_bmas"], gt_soft_labels, is_soft_targets=True
            ).item()

    return metrics


def evaluate_on_bregman(
    model,
    estimates,
    targets,
    is_same_task,
    is_soft_labels,
    args,
    upstream_is_soft_labels,
):
    is_mixed = upstream_is_soft_labels is not None

    # For Bregman, both datasets need to be soft
    if is_mixed:
        is_soft_labels = is_soft_labels and upstream_is_soft_labels

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed else ""

    if is_same_task and not isinstance(model, MCInfoNCEWrapper):
        gt_predictives_bregman_fbar = targets["gt_predictives_bregman_fbar"]
        gt_predictives_bregman_bma = targets["gt_predictives_bregman_bma"]

        gt_total_predictives_bregman_fbar = targets["gt_total_predictives_bregman_fbar"]
        gt_total_predictives_bregman_bma = targets["gt_total_predictives_bregman_bma"]

        if is_soft_labels:
            gt_biases_bregman_fbar = targets["gt_biases_bregman_fbar"]
            gt_biases_bregman_bma = targets["gt_biases_bregman_bma"]

    if is_soft_labels:
        gt_aleatorics_bregman = targets["gt_aleatorics_bregman"]

    if not isinstance(model, MCInfoNCEWrapper):
        gt_epistemics_bregman = targets["gt_epistemics_bregman"]

    for estimator_name in estimates:
        estimate = estimates[estimator_name]

        if not isinstance(model, MCInfoNCEWrapper):
            metrics[
                f"{key_prefix}{estimator_name}_rank_correlation_bregman_eu"
            ] = float(spearmanr(estimate, gt_epistemics_bregman)[0])
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_eu"] = (
                (estimate - gt_epistemics_bregman).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_eu"] = (
                (estimate - gt_epistemics_bregman).abs().mean().item()
            )

        if is_soft_labels:
            metrics[
                f"{key_prefix}{estimator_name}_rank_correlation_bregman_au"
            ] = float(spearmanr(estimate, gt_aleatorics_bregman)[0])
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_au"] = (
                (estimate - gt_aleatorics_bregman).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_au"] = (
                (estimate - gt_aleatorics_bregman).abs().mean().item()
            )

        if is_same_task and not isinstance(model, MCInfoNCEWrapper):
            metrics[
                f"{key_prefix}{estimator_name}_rank_correlation_bregman_pu_fbar"
            ] = float(spearmanr(estimate, gt_predictives_bregman_fbar)[0])
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_pu_fbar"] = (
                (estimate - gt_predictives_bregman_fbar).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_pu_fbar"] = (
                (estimate - gt_predictives_bregman_fbar).abs().mean().item()
            )

            metrics[
                f"{key_prefix}{estimator_name}_rank_correlation_bregman_total_pu_fbar"
            ] = float(spearmanr(estimate, gt_total_predictives_bregman_fbar)[0])
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_total_pu_fbar"] = (
                (estimate - gt_total_predictives_bregman_fbar).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_total_pu_fbar"] = (
                (estimate - gt_total_predictives_bregman_fbar).abs().mean().item()
            )

            metrics[
                f"{key_prefix}{estimator_name}_rank_correlation_bregman_pu_bma"
            ] = float(spearmanr(estimate, gt_predictives_bregman_bma)[0])
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_pu_bma"] = (
                (estimate - gt_predictives_bregman_bma).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_pu_bma"] = (
                (estimate - gt_predictives_bregman_bma).abs().mean().item()
            )

            metrics[
                f"{key_prefix}{estimator_name}_rank_correlation_bregman_total_pu_bma"
            ] = float(spearmanr(estimate, gt_total_predictives_bregman_bma)[0])
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_total_pu_bma"] = (
                (estimate - gt_total_predictives_bregman_bma).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_total_pu_bma"] = (
                (estimate - gt_total_predictives_bregman_bma).abs().mean().item()
            )

            if is_soft_labels:
                metrics[
                    f"{key_prefix}{estimator_name}_rank_correlation_bregman_b_fbar"
                ] = float(spearmanr(estimate, gt_biases_bregman_fbar)[0])
                metrics[f"{key_prefix}{estimator_name}_mse_bregman_b_fbar"] = (
                    (estimate - gt_biases_bregman_fbar).square().mean().item()
                )
                metrics[f"{key_prefix}{estimator_name}_mae_bregman_b_fbar"] = (
                    (estimate - gt_biases_bregman_fbar).abs().mean().item()
                )

                metrics[
                    f"{key_prefix}{estimator_name}_rank_correlation_bregman_b_bma"
                ] = float(spearmanr(estimate, gt_biases_bregman_bma)[0])
                metrics[f"{key_prefix}{estimator_name}_mse_bregman_b_bma"] = (
                    (estimate - gt_biases_bregman_bma).square().mean().item()
                )
                metrics[f"{key_prefix}{estimator_name}_mae_bregman_b_bma"] = (
                    (estimate - gt_biases_bregman_bma).abs().mean().item()
                )

    return metrics


def evaluate_on_correlation_of_decompositions(
    model,
    estimates,
    targets,
    is_same_task,
    is_soft_labels,
    args,
    upstream_is_soft_labels,
):
    is_evaluate_gt = args.is_evaluate_gt

    is_mixed = upstream_is_soft_labels is not None

    # For Bregman, both datasets need to be soft
    if is_mixed:
        is_soft_labels = is_soft_labels and upstream_is_soft_labels

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed else ""

    if not isinstance(model, MCInfoNCEWrapper):
        # BMA decomposition
        entropies_of_bma = estimates["entropies_of_bma"]
        expected_entropies = estimates["expected_entropies"]
        jensen_shannon_divergences = estimates["jensen_shannon_divergences"]

        metrics[f"{key_prefix}rank_correlation_bma_au_eu"] = float(
            spearmanr(expected_entropies, jensen_shannon_divergences)[0]
        )
        metrics[f"{key_prefix}correlation_bma_au_eu"] = float(
            pearsonr(expected_entropies, jensen_shannon_divergences)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bma_au_pu"] = float(
            spearmanr(expected_entropies, entropies_of_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bma_au_pu"] = float(
            pearsonr(expected_entropies, entropies_of_bma)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bma_eu_pu"] = float(
            spearmanr(jensen_shannon_divergences, entropies_of_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bma_eu_pu"] = float(
            pearsonr(jensen_shannon_divergences, entropies_of_bma)[0]
        )

        # Bregman decomposition estimates
        expected_divergences = estimates["expected_divergences"]
        expected_entropies_plus_expected_divergences = estimates[
            "expected_entropies_plus_expected_divergences"
        ]

        metrics[f"{key_prefix}rank_correlation_bregman_eu_au_hat"] = float(
            spearmanr(expected_divergences, expected_entropies)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_eu_au_hat"] = float(
            pearsonr(expected_divergences, expected_entropies)[0]
        )
        metrics[f"{key_prefix}rank_correlation_bregman_eu_pu_hat"] = float(
            spearmanr(
                expected_divergences, expected_entropies_plus_expected_divergences
            )[0]
        )
        metrics[f"{key_prefix}correlation_bregman_eu_pu_hat"] = float(
            pearsonr(
                expected_divergences, expected_entropies_plus_expected_divergences
            )[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_hat_pu_hat"] = float(
            spearmanr(expected_entropies, expected_entropies_plus_expected_divergences)[
                0
            ]
        )
        metrics[f"{key_prefix}correlation_bregman_au_hat_pu_hat"] = float(
            pearsonr(expected_entropies, expected_entropies_plus_expected_divergences)[
                0
            ]
        )

    if isinstance(model, DDUWrapper):
        ddu_aleatoric = estimates["expected_entropies"]
        ddu_epistemic = estimates["gmm_neg_log_densities"]
        metrics[f"{key_prefix}correlation_ddu_au_eu"] = float(
            pearsonr(ddu_aleatoric, ddu_epistemic)[0]
        )

    if not is_evaluate_gt or isinstance(model, MCInfoNCEWrapper):
        return metrics

    # Bregman decomposition GTs
    if is_same_task:
        gt_predictives_bregman_fbar = targets["gt_predictives_bregman_fbar"]
        gt_predictives_bregman_bma = targets["gt_predictives_bregman_bma"]

        gt_total_predictives_bregman_fbar = targets["gt_total_predictives_bregman_fbar"]
        gt_total_predictives_bregman_bma = targets["gt_total_predictives_bregman_bma"]

        if is_soft_labels:
            gt_biases_bregman_fbar = targets["gt_biases_bregman_fbar"]
            gt_biases_bregman_bma = targets["gt_biases_bregman_bma"]

    if is_soft_labels:
        gt_aleatorics_bregman = targets["gt_aleatorics_bregman"]

    gt_epistemics_bregman = targets["gt_epistemics_bregman"]

    can_evaluate_au_eu = is_soft_labels
    can_evaluate_au_b = can_evaluate_au_pu = can_evaluate_b_pu = (
        is_same_task and is_soft_labels
    )
    can_evaluate_eu_b = can_evaluate_au_b
    can_evaluate_eu_pu = is_same_task

    if can_evaluate_au_eu:
        metrics[f"{key_prefix}rank_correlation_bregman_au_eu"] = float(
            spearmanr(gt_aleatorics_bregman, gt_epistemics_bregman)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_au_eu"] = float(
            pearsonr(gt_aleatorics_bregman, gt_epistemics_bregman)[0]
        )

    if can_evaluate_au_b:
        metrics[f"{key_prefix}rank_correlation_bregman_au_b_fbar"] = float(
            spearmanr(gt_aleatorics_bregman, gt_biases_bregman_fbar)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_au_b_fbar"] = float(
            pearsonr(gt_aleatorics_bregman, gt_biases_bregman_fbar)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_b_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_biases_bregman_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_au_b_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_biases_bregman_bma)[0]
        )

    if can_evaluate_au_pu:
        metrics[f"{key_prefix}rank_correlation_bregman_au_pu_fbar"] = float(
            spearmanr(gt_aleatorics_bregman, gt_predictives_bregman_fbar)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_au_pu_fbar"] = float(
            pearsonr(gt_aleatorics_bregman, gt_predictives_bregman_fbar)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_pu_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_predictives_bregman_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_au_pu_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_predictives_bregman_bma)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_total_pu_fbar"] = float(
            spearmanr(gt_aleatorics_bregman, gt_total_predictives_bregman_fbar)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_au_total_pu_fbar"] = float(
            pearsonr(gt_aleatorics_bregman, gt_total_predictives_bregman_fbar)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_total_pu_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_total_predictives_bregman_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_au_total_pu_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_total_predictives_bregman_bma)[0]
        )

    if can_evaluate_b_pu:
        metrics[f"{key_prefix}rank_correlation_bregman_b_pu_fbar"] = float(
            spearmanr(gt_biases_bregman_fbar, gt_predictives_bregman_fbar)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_b_pu_fbar"] = float(
            pearsonr(gt_biases_bregman_fbar, gt_predictives_bregman_fbar)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_pu_bma"] = float(
            spearmanr(gt_biases_bregman_bma, gt_predictives_bregman_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_b_pu_bma"] = float(
            pearsonr(gt_biases_bregman_bma, gt_predictives_bregman_bma)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_total_pu_fbar"] = float(
            spearmanr(gt_biases_bregman_fbar, gt_total_predictives_bregman_fbar)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_b_total_pu_fbar"] = float(
            pearsonr(gt_biases_bregman_fbar, gt_total_predictives_bregman_fbar)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_total_pu_bma"] = float(
            spearmanr(gt_biases_bregman_bma, gt_total_predictives_bregman_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_b_total_pu_bma"] = float(
            pearsonr(gt_biases_bregman_bma, gt_total_predictives_bregman_bma)[0]
        )

    if can_evaluate_eu_b:
        metrics[f"{key_prefix}rank_correlation_bregman_eu_b_fbar"] = float(
            spearmanr(gt_epistemics_bregman, gt_biases_bregman_fbar)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_eu_b_fbar"] = float(
            pearsonr(gt_epistemics_bregman, gt_biases_bregman_fbar)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_eu_b_bma"] = float(
            spearmanr(gt_epistemics_bregman, gt_biases_bregman_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_eu_b_bma"] = float(
            pearsonr(gt_epistemics_bregman, gt_biases_bregman_bma)[0]
        )

    if can_evaluate_eu_pu:
        metrics[f"{key_prefix}rank_correlation_bregman_eu_pu_fbar"] = float(
            spearmanr(gt_epistemics_bregman, gt_predictives_bregman_fbar)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_eu_pu_fbar"] = float(
            pearsonr(gt_epistemics_bregman, gt_predictives_bregman_fbar)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_eu_pu_bma"] = float(
            spearmanr(gt_epistemics_bregman, gt_predictives_bregman_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_eu_pu_bma"] = float(
            pearsonr(gt_epistemics_bregman, gt_predictives_bregman_bma)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_eu_total_pu_fbar"] = float(
            spearmanr(gt_epistemics_bregman, gt_total_predictives_bregman_fbar)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_eu_total_pu_fbar"] = float(
            pearsonr(gt_epistemics_bregman, gt_total_predictives_bregman_fbar)[0]
        )

        metrics[f"{key_prefix}rank_correlation_bregman_eu_total_pu_bma"] = float(
            spearmanr(gt_epistemics_bregman, gt_total_predictives_bregman_bma)[0]
        )
        metrics[f"{key_prefix}correlation_bregman_eu_total_pu_bma"] = float(
            pearsonr(gt_epistemics_bregman, gt_total_predictives_bregman_bma)[0]
        )

    return metrics


def calculate_auroc(estimate, correctness, args, soft):
    if not soft:
        return auroc(estimate, correctness)

    num_repetitions = args.num_repetitions
    num_positives = (correctness * num_repetitions).round().long()
    comparison_range = (
        torch.arange(num_repetitions).unsqueeze(0).expand(correctness.shape[0], -1)
    )
    unrolled_labels = (comparison_range < num_positives.unsqueeze(1)).long().flatten()
    expanded_estimate = estimate.repeat_interleave(num_repetitions)

    return auroc(expanded_estimate, unrolled_labels)


def get_bundle(
    model,
    loader,
    device,
    amp_autocast,
    is_soft_labels,
    is_same_task,
    args,
):
    estimates = {}
    log_probs = {}
    targets = {}
    times = {}

    num_samples = len(loader.dataset)  # Total number of samples

    ### Ground truth containers

    ## Practical tasks

    # Abstained prediction
    # - needs validation loader
    # - but at least it's theoretically possible

    # For abstinence, correctness is calculated differently
    label_shape = next(iter(loader))[1].shape
    if is_same_task and is_soft_labels:
        assert label_shape[-1] == model.num_classes

    gt_hard_labels = torch.empty(num_samples, dtype=torch.long)
    targets["gt_hard_labels"] = gt_hard_labels

    # Correctness of prediction
    # We calculate the correctness in the CoP evaluation function. Here,
    # we only record the GT labels and our predictions (logits).
    # The reason is that there are two possible ways to treat soft labels

    # OOD detection
    # We don't record OOD-ness, as these labels are decided at a later point of the code

    # Proper scoring and calibration
    # We only need the labels and the logits to calculate these metrics

    ## Theoretical tasks

    if is_soft_labels:
        gt_soft_labels = torch.empty(num_samples, label_shape[1])
        targets["gt_soft_labels"] = gt_soft_labels

        # Aleatoric uncertainty (Bregman)
        gt_aleatorics_bregman = torch.empty(num_samples)
        targets["gt_aleatorics_bregman"] = gt_aleatorics_bregman
        # Also interested in how well the GT solves the practical tasks
        estimates["gt_aleatorics_bregman"] = gt_aleatorics_bregman

        # Bias (Bregman)
        if not isinstance(model, MCInfoNCEWrapper):
            gt_biases_bregman_fbar = torch.empty(num_samples)
            targets["gt_biases_bregman_fbar"] = gt_biases_bregman_fbar
            estimates["gt_biases_bregman_fbar"] = gt_biases_bregman_fbar

            gt_biases_bregman_bma = torch.empty(num_samples)
            targets["gt_biases_bregman_bma"] = gt_biases_bregman_bma
            estimates["gt_biases_bregman_bma"] = gt_biases_bregman_bma

    ### Estimate containers
    features = torch.empty(num_samples, model.num_features)

    if not isinstance(model, MCInfoNCEWrapper):
        # Predictive uncertainty (Bregman)
        gt_predictives_bregman_fbar = torch.empty(num_samples)
        targets["gt_predictives_bregman_fbar"] = gt_predictives_bregman_fbar
        estimates["gt_predictives_bregman_fbar"] = gt_predictives_bregman_fbar

        gt_total_predictives_bregman_fbar = torch.empty(num_samples)
        targets["gt_total_predictives_bregman_fbar"] = gt_total_predictives_bregman_fbar
        estimates[
            "gt_total_predictives_bregman_fbar"
        ] = gt_total_predictives_bregman_fbar

        gt_predictives_bregman_bma = torch.empty(num_samples)
        targets["gt_predictives_bregman_bma"] = gt_predictives_bregman_bma
        estimates["gt_predictives_bregman_bma"] = gt_predictives_bregman_bma

        gt_total_predictives_bregman_bma = torch.empty(num_samples)
        targets["gt_total_predictives_bregman_bma"] = gt_total_predictives_bregman_bma
        estimates["gt_total_predictives_bregman_bma"] = gt_total_predictives_bregman_bma

        # Epistemic uncertainty (Bregman)
        gt_epistemics_bregman = torch.empty(num_samples)
        targets["gt_epistemics_bregman"] = gt_epistemics_bregman

        # Time
        time_expected_entropy_m = AverageMeter()
        times["time_expected_entropy_m"] = time_expected_entropy_m
        time_expected_max_prob_m = AverageMeter()
        times["time_expected_max_prob_m"] = time_expected_max_prob_m
        time_entropy_of_bma_m = AverageMeter()
        times["time_entropy_of_bma_m"] = time_entropy_of_bma_m
        time_entropy_of_fbar_m = AverageMeter()
        times["time_entropy_of_fbar_m"] = time_entropy_of_fbar_m
        time_max_prob_of_bma_m = AverageMeter()
        times["time_max_prob_of_bma_m"] = time_max_prob_of_bma_m
        time_max_prob_of_fbar_m = AverageMeter()
        times["time_max_prob_of_fbar_m"] = time_max_prob_of_fbar_m
        time_expected_divergence_m = AverageMeter()
        times["time_expected_divergence_m"] = time_expected_divergence_m
        time_jsd_m = AverageMeter()
        times["time_jsd_m"] = time_jsd_m
        time_dempster_shafer_value_m = AverageMeter()
        times["time_dempster_shafer_value_m"] = time_dempster_shafer_value_m

        log_fbars = torch.empty(num_samples, model.num_classes)
        log_probs["log_fbars"] = log_fbars

        log_bmas = torch.empty(num_samples, model.num_classes)
        log_probs["log_bmas"] = log_bmas

        # AU
        expected_entropies = torch.empty(num_samples)
        estimates["expected_entropies"] = expected_entropies
        one_minus_expected_max_probs = torch.empty(num_samples)
        estimates["one_minus_expected_max_probs"] = one_minus_expected_max_probs

        # PU
        entropies_of_bma = torch.empty(num_samples)
        estimates["entropies_of_bma"] = entropies_of_bma
        entropies_of_fbar = torch.empty(num_samples)  # Just an extra thing to try out
        estimates["entropies_of_fbar"] = entropies_of_fbar
        one_minus_max_probs_of_bma = torch.empty(num_samples)
        estimates["one_minus_max_probs_of_bma"] = one_minus_max_probs_of_bma
        one_minus_max_probs_of_fbar = torch.empty(
            num_samples
        )  # Just an extra thing to try out
        estimates["one_minus_max_probs_of_fbar"] = one_minus_max_probs_of_fbar
        expected_entropies_plus_expected_divergences = torch.empty(num_samples)
        estimates[
            "expected_entropies_plus_expected_divergences"
        ] = expected_entropies_plus_expected_divergences

        # EU
        dempster_shafer_values = torch.empty(num_samples)
        estimates["dempster_shafer_values"] = dempster_shafer_values
        # Just a duplicate
        estimates["expected_divergences"] = gt_epistemics_bregman
        jensen_shannon_divergences = torch.empty(num_samples)
        estimates["jensen_shannon_divergences"] = jensen_shannon_divergences

        # This class gives "logits" that are different from the baseline model.
        if isinstance(model, NonIsotropicvMFWrapper):
            time_nivmf_inverse_kappa_m = AverageMeter()
            times["time_nivmf_inverse_kappa_m"] = time_nivmf_inverse_kappa_m

            nivmf_inverse_kappas = torch.empty(num_samples)
            estimates["nivmf_inverse_kappas"] = nivmf_inverse_kappas
        # This class modifies the model when it's not frozen, leading to different
        # logits.
        elif isinstance(model, BaseLossPredictionWrapper):
            # PU
            time_risk_value_m = AverageMeter()
            times["time_risk_value_m"] = time_risk_value_m
            risk_values = torch.empty(num_samples)
            estimates["risk_values"] = risk_values
        elif isinstance(model, DDUWrapper):
            time_gmm_neg_log_density_m = AverageMeter()
            times["time_gmm_neg_log_density_m"] = time_gmm_neg_log_density_m
            gmm_neg_log_densities = torch.empty(num_samples)
            estimates["gmm_neg_log_densities"] = gmm_neg_log_densities
        # This class also modifies the model when it's not frozen.
        elif isinstance(model, BaseCorrectnessPredictionWrapper):
            # PU
            time_error_probability_m = AverageMeter()
            times["time_error_probability_m"] = time_error_probability_m
            error_probabilities = torch.empty(num_samples)
            estimates["error_probabilities"] = error_probabilities
        # This class gives "logits" that are different from the baseline model.
        elif isinstance(model, DUQWrapper):
            # EU
            time_duq_value_m = AverageMeter()
            times["time_duq_value_m"] = time_duq_value_m
            duq_values = torch.empty(num_samples)
            estimates["duq_values"] = duq_values
        # While this class returns logits, it's post-hoc. As such, the logits are not
        # changed compared to the baseline model, so we'd get the same results.
        elif isinstance(model, MahalanobisWrapper):
            # EU
            time_mahalanobis_value_m = AverageMeter()
            times["time_mahalanobis_value_m"] = time_mahalanobis_value_m
            mahalanobis_values = torch.empty(num_samples)
            estimates["mahalanobis_values"] = mahalanobis_values
    # This class doesn't return any logits.
    else:
        time_mcinfonce_inverse_kappa_m = AverageMeter()
        times["time_mcinfonce_inverse_kappa_m"] = time_mcinfonce_inverse_kappa_m

        mcinfonce_inverse_kappas = torch.empty(num_samples)
        estimates["mcinfonce_inverse_kappas"] = mcinfonce_inverse_kappas

    if not isinstance(model, DeepEnsembleWrapper):
        current_ind = 0
        for input, label in loader:
            indices = slice(current_ind, current_ind + input.shape[0])

            if args.no_prefetcher:
                input = input.to(device)
                label = label.to(device)

            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            batch_size = input.shape[0]

            base_time_start = time.perf_counter()
            with amp_autocast():
                inference_dict = model(input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            base_time_end = time.perf_counter()
            base_time = base_time_end - base_time_start

            for key in list(inference_dict.keys()):
                inference_dict[key] = inference_dict[key].detach().cpu().float()

            inference_dict = convert_inference_dict(
                model, inference_dict, base_time, args
            )

            features[indices] = inference_dict["feature"]

            if not isinstance(
                model,
                MCInfoNCEWrapper,
            ):
                update_logit_based(
                    inference_dict=inference_dict,
                    indices=indices,
                    batch_size=batch_size,
                    log_fbars=log_fbars,
                    log_bmas=log_bmas,
                    gt_epistemics_bregman=gt_epistemics_bregman,
                    time_expected_entropy_m=time_expected_entropy_m,
                    time_expected_max_prob_m=time_expected_max_prob_m,
                    time_entropy_of_bma_m=time_entropy_of_bma_m,
                    time_entropy_of_fbar_m=time_entropy_of_fbar_m,
                    time_max_prob_of_bma_m=time_max_prob_of_bma_m,
                    time_max_prob_of_fbar_m=time_max_prob_of_fbar_m,
                    time_expected_divergence_m=time_expected_divergence_m,
                    time_jsd_m=time_jsd_m,
                    time_dempster_shafer_value_m=time_dempster_shafer_value_m,
                    expected_entropies=expected_entropies,
                    expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
                    one_minus_expected_max_probs=one_minus_expected_max_probs,
                    entropies_of_bma=entropies_of_bma,
                    entropies_of_fbar=entropies_of_fbar,
                    one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
                    one_minus_max_probs_of_fbar=one_minus_max_probs_of_fbar,
                    jensen_shannon_divergences=jensen_shannon_divergences,
                    dempster_shafer_values=dempster_shafer_values,
                )

            if isinstance(model, NonIsotropicvMFWrapper):
                update_nivmf(
                    inference_dict=inference_dict,
                    indices=indices,
                    batch_size=batch_size,
                    time_nivmf_inverse_kappa_m=time_nivmf_inverse_kappa_m,
                    nivmf_inverse_kappas=nivmf_inverse_kappas,
                )
            elif isinstance(model, BaseLossPredictionWrapper):
                update_losspred(
                    inference_dict=inference_dict,
                    indices=indices,
                    batch_size=batch_size,
                    time_risk_value_m=time_risk_value_m,
                    risk_values=risk_values,
                )
            elif isinstance(model, DDUWrapper):
                update_ddu(
                    inference_dict=inference_dict,
                    indices=indices,
                    batch_size=batch_size,
                    time_gmm_neg_log_density_m=time_gmm_neg_log_density_m,
                    gmm_neg_log_densities=gmm_neg_log_densities,
                )
            elif isinstance(model, BaseCorrectnessPredictionWrapper):
                update_corrpred(
                    inference_dict=inference_dict,
                    indices=indices,
                    batch_size=batch_size,
                    time_error_probability_m=time_error_probability_m,
                    error_probabilities=error_probabilities,
                )
            elif isinstance(model, DUQWrapper):
                update_duq(
                    inference_dict=inference_dict,
                    indices=indices,
                    batch_size=batch_size,
                    time_duq_value_m=time_duq_value_m,
                    duq_values=duq_values,
                )
            elif isinstance(model, MahalanobisWrapper):
                update_mahalanobis(
                    inference_dict=inference_dict,
                    indices=indices,
                    batch_size=batch_size,
                    time_mahalanobis_value_m=time_mahalanobis_value_m,
                    mahalanobis_values=mahalanobis_values,
                )
            elif isinstance(model, MCInfoNCEWrapper):
                update_mcinfonce(
                    inference_dict=inference_dict,
                    indices=indices,
                    batch_size=batch_size,
                    time_mcinfonce_inverse_kappa_m=time_mcinfonce_inverse_kappa_m,
                    mcinfonce_inverse_kappas=mcinfonce_inverse_kappas,
                )

            # GT containers
            if is_soft_labels:
                prob = label.float() / label.sum(dim=1, keepdim=True)  # Normalization
                prob = prob.cpu()
                gt_aleatorics_bregman[indices] = entropy(prob)

            if is_same_task:
                if not isinstance(model, MCInfoNCEWrapper):
                    log_fbar = inference_dict["log_fbar"]
                    log_bma = inference_dict["log_bma"]
                if is_soft_labels:
                    if not isinstance(model, MCInfoNCEWrapper):
                        log_prob = prob.log()
                        min_real = torch.finfo(log_prob.dtype).min
                        log_prob = torch.clamp(log_prob, min=min_real)

                        gt_biases_bregman_fbar[indices] = kl_divergence(
                            log_prob, log_fbar
                        )
                        gt_biases_bregman_bma[indices] = kl_divergence(
                            log_prob, log_bma
                        )
                        gt_predictives_bregman_fbar[indices] = (
                            gt_aleatorics_bregman[indices]
                            + gt_biases_bregman_fbar[indices]
                        )
                        gt_predictives_bregman_bma[indices] = (
                            gt_aleatorics_bregman[indices]
                            + gt_biases_bregman_bma[indices]
                        )
                        gt_total_predictives_bregman_fbar[indices] = (
                            gt_aleatorics_bregman[indices]
                            + gt_biases_bregman_fbar[indices]
                            + gt_epistemics_bregman[indices]
                        )
                        gt_total_predictives_bregman_bma[indices] = (
                            gt_aleatorics_bregman[indices]
                            + gt_biases_bregman_bma[indices]
                            + gt_epistemics_bregman[indices]
                        )
                    gt_soft_labels[indices] = prob
                    gt_hard_labels[indices] = prob.argmax(dim=1)
                else:
                    gt_hard_labels[indices] = label.cpu()

                    if not isinstance(model, MCInfoNCEWrapper):
                        gt_predictives_bregman_fbar[indices] = F.cross_entropy(
                            log_fbar, label.cpu()
                        )
                        gt_predictives_bregman_bma[indices] = F.cross_entropy(
                            log_bma, label.cpu()
                        )
                        gt_total_predictives_bregman_fbar[indices] = F.cross_entropy(
                            log_fbar, label.cpu()
                        )
                        gt_total_predictives_bregman_bma[indices] = F.cross_entropy(
                            log_bma, label.cpu()
                        )

            current_ind += input.shape[0]
    else:
        temp_logits = torch.empty(num_samples, model.num_models, model.num_classes)
        temp_features = torch.empty(num_samples, model.num_models, model.num_features)
        base_time_m = AverageMeter()

        for index in range(model.num_models):
            model.load_model(index)
            model.to(device=device)
            if args.channels_last:
                model.to(memory_format=torch.channels_last)

            current_ind = 0
            for input, label in loader:
                batch_size = input.shape[0]
                indices = slice(current_ind, current_ind + batch_size)

                if args.no_prefetcher:
                    input = input.to(device)
                    label = label.to(device)

                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                base_time_start = time.perf_counter()
                with amp_autocast():
                    inference_dict = model(input)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                base_time_end = time.perf_counter()
                base_time = base_time_end - base_time_start
                base_time_m.update(base_time, batch_size)

                temp_logits[indices, index, :] = inference_dict["logit"]
                temp_features[indices, index, :] = inference_dict["feature"]

                current_ind += batch_size

        # Aggregate logits and features
        avg_base_time = base_time_m.avg

        features = temp_features.mean(dim=1)

        current_ind = 0
        for input, label in loader:
            batch_size = input.shape[0]
            indices = slice(current_ind, current_ind + batch_size)

            inference_dict = {
                "logit": temp_logits[indices],
                "feature": features[indices],
            }

            inference_dict = convert_inference_dict(
                model,
                inference_dict,
                avg_base_time,
                args,
            )

            update_logit_based(
                inference_dict=inference_dict,
                indices=indices,
                batch_size=batch_size,
                log_fbars=log_fbars,
                log_bmas=log_bmas,
                gt_epistemics_bregman=gt_epistemics_bregman,
                time_expected_entropy_m=time_expected_entropy_m,
                time_expected_max_prob_m=time_expected_max_prob_m,
                time_entropy_of_bma_m=time_entropy_of_bma_m,
                time_entropy_of_fbar_m=time_entropy_of_fbar_m,
                time_max_prob_of_bma_m=time_max_prob_of_bma_m,
                time_max_prob_of_fbar_m=time_max_prob_of_fbar_m,
                time_expected_divergence_m=time_expected_divergence_m,
                time_jsd_m=time_jsd_m,
                expected_entropies=expected_entropies,
                expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
                one_minus_expected_max_probs=one_minus_expected_max_probs,
                entropies_of_bma=entropies_of_bma,
                entropies_of_fbar=entropies_of_fbar,
                one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
                one_minus_max_probs_of_fbar=one_minus_max_probs_of_fbar,
                jensen_shannon_divergences=jensen_shannon_divergences,
                dempster_shafer_values=dempster_shafer_values,
            )

            # GT containers
            if is_soft_labels:
                prob = label.float() / label.sum(dim=1, keepdim=True)  # Normalization
                prob = prob.cpu()
                gt_aleatorics_bregman[indices] = entropy(prob)

            if is_same_task:
                log_fbar = inference_dict["log_fbar"]
                log_bma = inference_dict["log_bma"]
                if is_soft_labels:
                    log_prob = prob.log()
                    min_real = torch.finfo(log_prob.dtype).min
                    log_prob = torch.clamp(log_prob, min=min_real)
                    gt_biases_bregman_fbar[indices] = kl_divergence(log_prob, log_fbar)
                    gt_biases_bregman_bma[indices] = kl_divergence(log_prob, log_bma)
                    gt_predictives_bregman_fbar[indices] = (
                        gt_aleatorics_bregman[indices] + gt_biases_bregman_fbar[indices]
                    )
                    gt_predictives_bregman_bma[indices] = (
                        gt_aleatorics_bregman[indices] + gt_biases_bregman_bma[indices]
                    )
                    gt_total_predictives_bregman_fbar[indices] = (
                        gt_aleatorics_bregman[indices]
                        + gt_biases_bregman_fbar[indices]
                        + gt_epistemics_bregman[indices]
                    )
                    gt_total_predictives_bregman_bma[indices] = (
                        gt_aleatorics_bregman[indices]
                        + gt_biases_bregman_bma[indices]
                        + gt_epistemics_bregman[indices]
                    )
                    gt_soft_labels[indices] = prob
                    gt_hard_labels[indices] = prob.argmax(dim=1)
                else:
                    label = label.cpu()
                    gt_hard_labels[indices] = label
                    gt_predictives_bregman_fbar[indices] = F.cross_entropy(
                        log_fbar, label
                    )
                    gt_predictives_bregman_bma[indices] = F.cross_entropy(
                        log_bma, label
                    )
                    gt_total_predictives_bregman_fbar[indices] = F.cross_entropy(
                        log_fbar, label
                    )
                    gt_total_predictives_bregman_bma[indices] = F.cross_entropy(
                        log_bma, label
                    )

            current_ind += batch_size

    # Calculate correctness indicators
    gt_zero_shot_correctnesses = recall_at_one(features, gt_hard_labels, mode="faiss")
    targets["gt_zero_shot_correctnesses"] = gt_zero_shot_correctnesses

    if is_same_task and not isinstance(model, MCInfoNCEWrapper):
        predicted_labels_fbar = log_probs["log_fbars"].argmax(dim=1)

        targets["gt_hard_fbar_correctnesses"] = predicted_labels_fbar.eq(
            targets["gt_hard_labels"]
        ).int()

        _, predicted_labels_fbar_top5 = torch.topk(log_probs["log_fbars"], 5, dim=1)
        expanded_gt_hard_labels = (
            targets["gt_hard_labels"]
            .unsqueeze(dim=1)
            .expand_as(predicted_labels_fbar_top5)
        )
        targets["gt_hard_fbar_correctnesses_top5"] = (
            predicted_labels_fbar_top5.eq(expanded_gt_hard_labels).max(dim=1)[0].int()
        )

        predicted_labels_bma = log_probs["log_bmas"].argmax(dim=1)

        targets["gt_hard_bma_correctnesses"] = predicted_labels_bma.eq(
            targets["gt_hard_labels"]
        ).int()

        _, predicted_labels_bma_top5 = torch.topk(log_probs["log_bmas"], 5, dim=1)
        targets["gt_hard_bma_correctnesses_top5"] = (
            predicted_labels_bma_top5.eq(expanded_gt_hard_labels).max(dim=1)[0].int()
        )

        if is_soft_labels:
            targets["gt_soft_fbar_correctnesses"] = (
                targets["gt_soft_labels"]
                .gather(dim=1, index=predicted_labels_fbar.unsqueeze(dim=1))
                .squeeze(dim=1)
            )

            indexed_gt_soft_labels_fbar = targets["gt_soft_labels"].gather(
                dim=1, index=predicted_labels_fbar_top5
            )
            targets[
                "gt_soft_fbar_correctnesses_top5"
            ] = indexed_gt_soft_labels_fbar.max(dim=1)[0]

            targets["gt_soft_bma_correctnesses"] = (
                targets["gt_soft_labels"]
                .gather(dim=1, index=predicted_labels_bma.unsqueeze(dim=1))
                .squeeze(dim=1)
            )

            indexed_gt_soft_labels_bma = targets["gt_soft_labels"].gather(
                dim=1, index=predicted_labels_bma_top5
            )
            targets["gt_soft_bma_correctnesses_top5"] = indexed_gt_soft_labels_bma.max(
                dim=1
            )[0]

    # Extract averages from the AverageMeters
    for key in list(times.keys()):
        times[key] = times[key].avg

    if is_soft_labels:
        faulty_indices = targets["gt_aleatorics_bregman"].isnan()

        if faulty_indices.sum() > 0:
            for estimator_name in list(estimates.keys()):
                estimates[estimator_name] = estimates[estimator_name][~faulty_indices]

            for log_prob_name in list(log_probs.keys()):
                log_probs[log_prob_name] = log_probs[log_prob_name][~faulty_indices]

            for target_name in list(targets.keys()):
                targets[target_name] = targets[target_name][~faulty_indices]

    return estimates, log_probs, targets, times


def convert_inference_dict(model, inference_dict, base_time, args):
    converted_inference_dict = {}

    features = inference_dict["feature"]
    converted_inference_dict["feature"] = features

    if not isinstance(model, MCInfoNCEWrapper):
        time_log_probs_start = time.perf_counter()

        min_real = torch.finfo(features.dtype).min

        if isinstance(model, DirichletWrapper):
            alphas = inference_dict["alpha"]  # [B, C]
            log_probs = (
                torch.distributions.Dirichlet(alphas)
                .sample((args.num_mc_samples,))
                .permute(1, 0, 2)
                .log()
                .clamp(min=min_real)
            )  # [B, S, C]

            time_log_probs_end = time.perf_counter()
            time_log_probs = time_log_probs_end - time_log_probs_start + base_time

            time_sum_alphas_start = time.perf_counter()
            sum_alphas = alphas.sum(dim=1)  # [B]
            time_sum_alphas_end = time.perf_counter()
            time_sum_alphas = time_sum_alphas_end - time_sum_alphas_start + base_time

            time_mean_alphas_start = time.perf_counter()
            mean_alphas = alphas.div(sum_alphas.unsqueeze(1))  # [B, C]
            time_mean_alphas_end = time.perf_counter()
            time_mean_alphas = (
                time_mean_alphas_end - time_mean_alphas_start + time_sum_alphas
            )

            log_bma = mean_alphas.log().clamp(min=min_real)
            converted_inference_dict["log_bma"] = log_bma

            time_log_fbar_start = time.perf_counter()
            log_fbar = F.log_softmax(log_probs.mean(dim=1), dim=-1)  # [B, C]
            time_log_fbar_end = time.perf_counter()
            time_log_fbar = time_log_fbar_end - time_log_fbar_start + time_log_probs
            converted_inference_dict["log_fbar"] = log_fbar

            time_expected_entropy_start = time.perf_counter()
            digamma_term = torch.digamma(alphas + 1) - torch.digamma(
                sum_alphas + 1
            ).unsqueeze(
                1
            )  # [B, C]
            expected_entropy = -mean_alphas.mul(digamma_term).sum(dim=1)  # [B]
            time_expected_entropy_end = time.perf_counter()
            time_expected_entropy = (
                time_expected_entropy_end
                - time_expected_entropy_start
                + time_mean_alphas
            )
            converted_inference_dict["expected_entropy"] = expected_entropy
            converted_inference_dict["time_expected_entropy"] = time_expected_entropy

            time_expected_divergence_start = time.perf_counter()
            expected_divergence = kl_divergence(
                log_fbar, log_probs.permute(1, 0, 2)
            ).mean(dim=0)
            time_expected_divergence_end = time.perf_counter()
            time_expected_divergence = (
                time_expected_divergence_end
                - time_expected_divergence_start
                + time_log_fbar
            )
            converted_inference_dict["expected_divergence"] = expected_divergence
            converted_inference_dict[
                "time_expected_divergence"
            ] = time_expected_divergence

            time_probs_start = time.perf_counter()
            probs = log_probs.exp()
            time_probs_end = time.perf_counter()
            time_probs = time_probs_end - time_probs_start

            time_expected_max_prob_start = time.perf_counter()
            expected_max_prob = probs.max(dim=-1)[0].mean(dim=1)
            time_expected_max_prob_end = time.perf_counter()
            time_expected_max_prob = (
                time_expected_max_prob_end - time_expected_max_prob_start + time_probs
            )
            converted_inference_dict["expected_max_prob"] = expected_max_prob
            converted_inference_dict["time_expected_max_prob"] = time_expected_max_prob

            time_entropy_of_bma_start = time.perf_counter()
            entropy_of_bma = entropy(mean_alphas)
            time_entropy_of_bma_end = time.perf_counter()
            time_entropy_of_bma = (
                time_entropy_of_bma_end - time_entropy_of_bma_start + time_mean_alphas
            )
            converted_inference_dict["entropy_of_bma"] = entropy_of_bma
            converted_inference_dict["time_entropy_of_bma"] = time_entropy_of_bma

            time_fbar_start = time.perf_counter()
            fbar = log_fbar.exp()
            time_fbar_end = time.perf_counter()
            time_fbar = time_fbar_end - time_fbar_start

            time_entropy_of_fbar_start = time.perf_counter()
            entropy_of_fbar = entropy(fbar)
            time_entropy_of_fbar_end = time.perf_counter()
            time_entropy_of_fbar = (
                time_entropy_of_fbar_end - time_entropy_of_fbar_start + time_fbar
            )
            converted_inference_dict["entropy_of_fbar"] = entropy_of_fbar
            converted_inference_dict["time_entropy_of_fbar"] = time_entropy_of_fbar

            time_max_prob_of_bma_start = time.perf_counter()
            max_prob_of_bma = mean_alphas.max(dim=-1)[0]
            time_max_prob_of_bma_end = time.perf_counter()
            time_max_prob_of_bma = (
                time_max_prob_of_bma_end - time_max_prob_of_bma_start + time_mean_alphas
            )
            converted_inference_dict["max_prob_of_bma"] = max_prob_of_bma
            converted_inference_dict["time_max_prob_of_bma"] = time_max_prob_of_bma

            time_max_prob_of_fbar_start = time.perf_counter()
            max_prob_of_fbar = fbar.max(dim=-1)[0]
            time_max_prob_of_fbar_end = time.perf_counter()
            time_max_prob_of_fbar = (
                time_max_prob_of_fbar_end - time_max_prob_of_fbar_start + time_fbar
            )
            converted_inference_dict["max_prob_of_fbar"] = max_prob_of_fbar
            converted_inference_dict["time_max_prob_of_fbar"] = time_max_prob_of_fbar

            time_jsd_start = time.perf_counter()
            jensen_shannon_divergence = entropy_of_bma - expected_entropy
            time_jsd_end = time.perf_counter()
            time_jsd = (
                time_jsd_end
                - time_jsd_start
                + time_entropy_of_bma
                + time_expected_entropy
                - time_probs
            )
            converted_inference_dict[
                "jensen_shannon_divergence"
            ] = jensen_shannon_divergence
            converted_inference_dict["time_jsd"] = time_jsd

            time_dempster_shafer_value_start = time.perf_counter()
            num_classes = alphas.shape[1]
            dempster_shafer_value = num_classes / sum_alphas  # [B]
            time_dempster_shafer_value_end = time.perf_counter()
            time_dempster_shafer_value = (
                time_dempster_shafer_value_end
                - time_dempster_shafer_value_start
                + time_sum_alphas
            )
            converted_inference_dict["dempster_shafer_value"] = dempster_shafer_value
            converted_inference_dict[
                "time_dempster_shafer_value"
            ] = time_dempster_shafer_value
        else:
            logits = inference_dict["logit"]
            if logits.dim() == 2:  # [B, C]
                logits = logits.unsqueeze(dim=1)  # [B, 1, C]
            log_probs = F.log_softmax(logits, dim=-1)  # [B, S, C]

            time_log_probs_end = time.perf_counter()
            time_log_probs = time_log_probs_end - time_log_probs_start + base_time

            time_probs_start = time.perf_counter()
            probs = log_probs.exp()  # [B, S, C]
            time_probs_end = time.perf_counter()
            time_probs = time_probs_end - time_probs_start + time_log_probs

            time_log_fbar_start = time.perf_counter()

            log_fbar = F.log_softmax(log_probs.mean(dim=1), dim=-1)  # [B, C]

            time_log_fbar_end = time.perf_counter()
            time_log_fbar = time_log_fbar_end - time_log_fbar_start + time_log_probs

            time_fbar_start = time.perf_counter()
            fbar = log_fbar.exp()
            time_fbar_end = time.perf_counter()
            time_fbar = time_fbar_end - time_fbar_start + time_log_fbar
            converted_inference_dict["log_fbar"] = log_fbar

            time_bma_start = time.perf_counter()
            bma = probs.mean(dim=1)  # [B, C]
            time_bma_end = time.perf_counter()
            time_bma = time_bma_end - time_bma_start + time_probs

            log_bma = bma.log()  # [B, C]
            log_bma = torch.clamp(log_bma, min=min_real)
            converted_inference_dict["log_bma"] = log_bma

            time_expected_entropy_start = time.perf_counter()
            expected_entropy = entropy(probs).mean(dim=-1)
            time_expected_entropy_end = time.perf_counter()
            time_expected_entropy = (
                time_expected_entropy_end - time_expected_entropy_start + time_probs
            )
            converted_inference_dict["expected_entropy"] = expected_entropy
            converted_inference_dict["time_expected_entropy"] = time_expected_entropy

            time_expected_divergence_start = time.perf_counter()
            expected_divergence = kl_divergence(
                log_fbar, log_probs.permute(1, 0, 2)
            ).mean(dim=0)
            time_expected_divergence_end = time.perf_counter()
            time_expected_divergence = (
                time_expected_divergence_end
                - time_expected_divergence_start
                + time_log_fbar
            )
            converted_inference_dict["expected_divergence"] = expected_divergence
            converted_inference_dict[
                "time_expected_divergence"
            ] = time_expected_divergence

            time_expected_max_prob_start = time.perf_counter()
            expected_max_prob = probs.max(dim=-1)[0].mean(dim=1)
            time_expected_max_prob_end = time.perf_counter()
            time_expected_max_prob = (
                time_expected_max_prob_end - time_expected_max_prob_start + time_probs
            )
            converted_inference_dict["expected_max_prob"] = expected_max_prob
            converted_inference_dict["time_expected_max_prob"] = time_expected_max_prob

            time_entropy_of_bma_start = time.perf_counter()
            entropy_of_bma = entropy(bma)
            time_entropy_of_bma_end = time.perf_counter()
            time_entropy_of_bma = (
                time_entropy_of_bma_end - time_entropy_of_bma_start + time_bma
            )
            converted_inference_dict["entropy_of_bma"] = entropy_of_bma
            converted_inference_dict["time_entropy_of_bma"] = time_entropy_of_bma

            time_entropy_of_fbar_start = time.perf_counter()
            entropy_of_fbar = entropy(fbar)
            time_entropy_of_fbar_end = time.perf_counter()
            time_entropy_of_fbar = (
                time_entropy_of_fbar_end - time_entropy_of_fbar_start + time_fbar
            )
            converted_inference_dict["entropy_of_fbar"] = entropy_of_fbar
            converted_inference_dict["time_entropy_of_fbar"] = time_entropy_of_fbar

            time_max_prob_of_bma_start = time.perf_counter()
            max_prob_of_bma = bma.max(dim=-1)[0]
            time_max_prob_of_bma_end = time.perf_counter()
            time_max_prob_of_bma = (
                time_max_prob_of_bma_end - time_max_prob_of_bma_start + time_bma
            )
            converted_inference_dict["max_prob_of_bma"] = max_prob_of_bma
            converted_inference_dict["time_max_prob_of_bma"] = time_max_prob_of_bma

            time_max_prob_of_fbar_start = time.perf_counter()
            max_prob_of_fbar = fbar.max(dim=-1)[0]
            time_max_prob_of_fbar_end = time.perf_counter()
            time_max_prob_of_fbar = (
                time_max_prob_of_fbar_end - time_max_prob_of_fbar_start + time_fbar
            )
            converted_inference_dict["max_prob_of_fbar"] = max_prob_of_fbar
            converted_inference_dict["time_max_prob_of_fbar"] = time_max_prob_of_fbar

            time_jsd_start = time.perf_counter()
            jensen_shannon_divergence = entropy_of_bma - expected_entropy
            time_jsd_end = time.perf_counter()
            time_jsd = (
                time_jsd_end
                - time_jsd_start
                + time_entropy_of_bma
                + time_expected_entropy
                - time_probs
            )
            converted_inference_dict[
                "jensen_shannon_divergence"
            ] = jensen_shannon_divergence
            converted_inference_dict["time_jsd"] = time_jsd

            time_dempster_shafer_value_start = time.perf_counter()
            dempster_shafer_value = dempster_shafer_metric(logits.mean(dim=1))
            time_dempster_shafer_value_end = time.perf_counter()
            time_dempster_shafer_value = (
                time_dempster_shafer_value_end
                - time_dempster_shafer_value_start
                + base_time
            )
            convert_inference_dict["dempster_shafer_value"] = dempster_shafer_value
            convert_inference_dict["time_dempster_shafer_value"] = dempster_shafer_value

        if isinstance(model, NonIsotropicvMFWrapper):
            converted_inference_dict["nivmf_inverse_kappa"] = inference_dict[
                "nivmf_inverse_kappa"
            ]
            converted_inference_dict["time_nivmf_inverse_kappa"] = base_time
        elif isinstance(model, BaseLossPredictionWrapper):
            converted_inference_dict["risk_value"] = inference_dict["risk_value"]
            converted_inference_dict["time_risk_value"] = base_time
        elif isinstance(model, DDUWrapper):
            converted_inference_dict["gmm_neg_log_density"] = inference_dict[
                "gmm_neg_log_density"
            ]
            converted_inference_dict["time_gmm_neg_log_density"] = base_time
        elif isinstance(model, BaseCorrectnessPredictionWrapper):
            converted_inference_dict["error_probability"] = inference_dict[
                "error_probability"
            ]
            converted_inference_dict["time_error_probability"] = base_time
        elif isinstance(model, DUQWrapper):
            converted_inference_dict["duq_value"] = inference_dict["duq_value"]
            converted_inference_dict["time_duq_value"] = base_time
        elif isinstance(model, MahalanobisWrapper):
            converted_inference_dict["mahalanobis_value"] = inference_dict[
                "mahalanobis_value"
            ]
            converted_inference_dict["time_mahalanobis_value"] = base_time
    else:
        converted_inference_dict["mcinfonce_inverse_kappa"] = inference_dict[
            "mcinfonce_inverse_kappa"
        ]
        converted_inference_dict["time_mcinfonce_inverse_kappa"] = base_time

    return converted_inference_dict


def update_logit_based(
    inference_dict,
    indices,
    batch_size,
    log_fbars,
    log_bmas,
    gt_epistemics_bregman,
    time_expected_entropy_m,
    time_expected_max_prob_m,
    time_entropy_of_bma_m,
    time_entropy_of_fbar_m,
    time_max_prob_of_bma_m,
    time_max_prob_of_fbar_m,
    time_expected_divergence_m,
    time_jsd_m,
    time_dempster_shafer_value_m,
    expected_entropies,
    expected_entropies_plus_expected_divergences,
    one_minus_expected_max_probs,
    entropies_of_bma,
    entropies_of_fbar,
    one_minus_max_probs_of_bma,
    one_minus_max_probs_of_fbar,
    jensen_shannon_divergences,
    dempster_shafer_values,
):
    log_fbars[indices] = inference_dict["log_fbar"]
    log_bmas[indices] = inference_dict["log_bma"]
    gt_epistemics_bregman[indices] = inference_dict["expected_divergence"]

    time_expected_entropy_m.update(inference_dict["time_expected_entropy"], batch_size)
    time_expected_max_prob_m.update(
        inference_dict["time_expected_max_prob"], batch_size
    )
    time_entropy_of_bma_m.update(inference_dict["time_entropy_of_bma"], batch_size)
    time_entropy_of_fbar_m.update(inference_dict["time_entropy_of_fbar"], batch_size)
    time_max_prob_of_bma_m.update(inference_dict["time_max_prob_of_bma"], batch_size)
    time_max_prob_of_fbar_m.update(inference_dict["time_max_prob_of_fbar"], batch_size)
    time_expected_divergence_m.update(
        inference_dict["time_expected_divergence"], batch_size
    )
    time_jsd_m.update(inference_dict["time_jsd"], batch_size)
    time_dempster_shafer_value_m.update(
        inference_dict["time_dempster_shafer_value"], batch_size
    )

    expected_entropies[indices] = inference_dict["expected_entropy"]
    expected_entropies_plus_expected_divergences[indices] = (
        expected_entropies[indices] + gt_epistemics_bregman[indices]
    )
    one_minus_expected_max_probs[indices] = 1 - inference_dict["expected_max_prob"]
    entropies_of_bma[indices] = inference_dict["entropy_of_bma"]
    entropies_of_fbar[indices] = inference_dict["entropy_of_fbar"]
    one_minus_max_probs_of_bma[indices] = 1 - inference_dict["max_prob_of_bma"]
    one_minus_max_probs_of_fbar[indices] = 1 - inference_dict["max_prob_of_fbar"]
    jensen_shannon_divergences[indices] = inference_dict["jensen_shannon_divergence"]
    dempster_shafer_values[indices] = inference_dict["dempster_shafer_value"]


def update_nivmf(
    inference_dict,
    indices,
    batch_size,
    time_nivmf_inverse_kappa_m,
    nivmf_inverse_kappas,
):
    time_nivmf_inverse_kappa_m.update(
        inference_dict["time_nivmf_inverse_kappa"], batch_size
    )
    nivmf_inverse_kappas[indices] = inference_dict["nivmf_inverse_kappa"]


def update_losspred(
    inference_dict, indices, batch_size, time_risk_value_m, risk_values
):
    time_risk_value_m.update(inference_dict["time_risk_value"], batch_size)
    risk_values[indices] = inference_dict["risk_value"]


def update_ddu(
    inference_dict,
    indices,
    batch_size,
    time_gmm_neg_log_density_m,
    gmm_neg_log_densities,
):
    time_gmm_neg_log_density_m.update(
        inference_dict["time_gmm_neg_log_density"], batch_size
    )
    gmm_neg_log_densities[indices] = inference_dict["gmm_neg_log_density"]


def update_corrpred(
    inference_dict, indices, batch_size, time_error_probability_m, error_probabilities
):
    time_error_probability_m.update(
        inference_dict["time_error_probability"], batch_size
    )
    error_probabilities[indices] = inference_dict["error_probability"]


def update_duq(inference_dict, indices, batch_size, time_duq_value_m, duq_values):
    time_duq_value_m.update(inference_dict["time_duq_value"], batch_size)
    duq_values[indices] = inference_dict["duq_value"]


def update_mahalanobis(
    inference_dict,
    indices,
    batch_size,
    time_mahalanobis_value_m,
    mahalanobis_values,
):
    time_mahalanobis_value_m.update(
        inference_dict["time_mahalanobis_value"], batch_size
    )
    mahalanobis_values[indices] = inference_dict["mahalanobis_value"]


def update_mcinfonce(
    inference_dict,
    indices,
    batch_size,
    time_mcinfonce_inverse_kappa_m,
    mcinfonce_inverse_kappas,
):
    time_mcinfonce_inverse_kappa_m.update(
        inference_dict["time_mcinfonce_inverse_kappa"], batch_size
    )
    mcinfonce_inverse_kappas[indices] = inference_dict["mcinfonce_inverse_kappa"]
