import os
import sys

import numpy as np
from tqdm import tqdm
import wandb
import json
import pandas as pd

from scipy.stats import spearmanr

sys.path.insert(0, "..")

from utils import ESTIMATOR_CONVERSION_DICT, ESTIMATORLESS_METRICS, CONSTRAINED_METRICS


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    with open("../../wandb_key.json") as f:
        wandb_key = json.load(f)["key"]

    wandb.login(key=wandb_key)
    api = wandb.Api()

    method_to_ids = {
        "GP": ("0zh85pjp", "82wiia5a"),
        "HET-XL": ("n6ocb8vt", "8pzbl9zq"),
        "CE Baseline": ("75316qay", "mj4gt28n"),
        "MC-Dropout": ("iphs7vdj", "nwmia4kf"),
        "SNGP": ("5l11sz1l", "b1dd9bjf"),
        "Shallow Ens.": ("50dvkkny", "k3v4wzua"),
        "Loss Pred.": ("qthh97bn", "t3j6wcsa"),
        "Corr. Pred.": ("7bexzi5z", "ymlbxdms"),
        "Deep Ens.": ("oyn8zlw5", "yw72v367"),
        "Laplace": ("i170wvxa", "7irimi02"),
        "Mahalanobis": ("iovcgd69", "aeb5oky6"),
        "Temperature": ("9mqh7if3", "5j5qcw9l"),
        "DDU": ("n5g7bnct", "ipcewyua"),
        "HET": ("7yusrr4s", "3l8nkci8"),
        "EDL": ("52ebshff", "ihcciqqt"),
        "PostNet": ("g9v0j4p4", "c3wpoy10"),
    }

    metric_dict = {
        "auroc_hard_bma_correctness_original": "Correctness AUROC",
        "ece_hard_bma_correctness_original": "ECE",
        "brier_score_hard_bma_correctness_original": "Correctness Brier",
        "log_prob_score_hard_bma_correctness_original": "Correctness Log Prob.",
        "hard_bma_raulc_original": "rAULC",
        "hard_bma_eaurc_original": "E-AURC",
        "cumulative_hard_bma_abstinence_auc_original": "AUAC",
        "hard_bma_accuracy_original": "Accuracy",
        "log_prob_score_hard_bma_aleatoric_original": "Aleatoric Log Prob.",
        "brier_score_hard_fbar_aleatoric_original": "Aleatoric Brier",
        "rank_correlation_bregman_au": "Aleatoric Rank Corr.",
        "auroc_multiple_labels": "Aleatoric AUROC",
        "auroc_oodness": "OOD AUROC",
    }

    performance_matrix_imagenet = np.zeros((len(metric_dict), len(method_to_ids)))
    performance_matrix_cifar = np.zeros((len(metric_dict), len(method_to_ids)))
    correlation_vector = np.zeros((len(metric_dict),))

    id_prefix = "best_id_test"
    mixture_prefix_cifar = "best_ood_test_soft/cifar10S2_mixed_soft/cifar10"
    mixture_prefix_imagenet = "best_ood_test_soft/imagenetS2_mixed_soft/imagenet"

    metric_names = []

    for j, (method_name, (imagenet_id, cifar_id)) in enumerate(
        tqdm(method_to_ids.items())
    ):
        sweep_imagenet = api.sweep(f"bmucsanyi/bias/{imagenet_id}")
        sweep_cifar = api.sweep(f"bmucsanyi/bias/{cifar_id}")

        for i, (metric_id, metric_name) in enumerate(metric_dict.items()):
            if j == 0:
                metric_names.append(metric_name)

            metric = {}

            for sweep, performance_matrix, prefix in zip(
                [sweep_imagenet, sweep_cifar],
                [performance_matrix_imagenet, performance_matrix_cifar],
                [mixture_prefix_imagenet, mixture_prefix_cifar],
            ):
                prefix = id_prefix if metric_name != "OOD AUROC" else prefix
                metric = {}
                for run in sweep.runs:
                    if run.state != "finished":
                        continue

                    for key in sorted(run.summary.keys()):
                        if key.startswith(prefix) and key.endswith(metric_id):
                            stripped_key = key.replace(f"{prefix}_", "").replace(
                                f"_{metric_id}", ""
                            )

                            if (
                                "mixed" in stripped_key
                                or "gt" in stripped_key
                                or run.summary[key] == "NaN"
                                or not (
                                    stripped_key in ESTIMATOR_CONVERSION_DICT
                                    or stripped_key in ESTIMATORLESS_METRICS
                                )
                            ):
                                continue

                            if stripped_key not in metric:
                                metric[stripped_key] = [run.summary[key]]
                            else:
                                metric[stripped_key].append(run.summary[key])

                            if metric_name in ["ECE", "E-AURC"]:
                                metric[stripped_key][-1] *= -1

                for key in tuple(metric.keys()):
                    metric[key] = np.mean(metric[key])

                if "brier_score_hard_fbar_aleatoric_original" in metric:
                    metric = {
                        "brier_score_hard_fbar_aleatoric_original": metric[
                            "brier_score_hard_fbar_aleatoric_original"
                        ]
                    }

                if (
                    metric_id not in ESTIMATORLESS_METRICS
                    and metric_id not in CONSTRAINED_METRICS
                ):
                    if method_name == "Corr. Pred.":
                        aggregated_key = "error_probabilities"
                    elif method_name == "Loss Pred.":
                        aggregated_key = "risk_values"
                    elif method_name == "Mahalanobis":
                        aggregated_key = "mahalanobis_values"
                    elif method_name == "DDU" and metric_id == "auroc_oodness":
                        aggregated_key = "gmm_neg_log_densities"
                    else:
                        aggregated_key = None
                else:
                    aggregated_key = None

                if aggregated_key is None:
                    aggregated_key = max(metric.items(), key=lambda x: x[1])[0]

                try:
                    performance_matrix[i, j] = metric[aggregated_key]
                except KeyError:
                    print(aggregated_key, metric_id)

    print("Perf matrix ImageNet:", performance_matrix_imagenet)
    print("Perf matrix CIFAR-10:", performance_matrix_cifar)

    best_methods_imagenet = np.argmax(performance_matrix_imagenet, axis=-1)
    best_methods_cifar = np.argmax(performance_matrix_cifar, axis=-1)

    print("Best methods on ImageNet:", best_methods_imagenet)
    print("Best methods on CIFAR-10:", best_methods_cifar)

    for i in range(len(metric_dict)):
        perf_imagenet = performance_matrix_imagenet[i, :]
        perf_cifar = performance_matrix_cifar[i, :]
        correlation_vector[i] = spearmanr(perf_imagenet, perf_cifar)[0]

    df = pd.DataFrame({"Metric": metric_names, "Rank Corr.": correlation_vector})
    print(df)


if __name__ == "__main__":
    main()
