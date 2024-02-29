import os
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import wandb

from tueplots import bundles
from scipy.stats import spearmanr

plt.rcParams.update(
    bundles.icml2022(family="serif", usetex=True, nrows=1, column="half")
)

plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


ESTIMATOR_CONVERSION_DICT = {
    "entropies_of_fbar": r"$\mathbb{H}(\bar{f})$",
    "entropies_of_bma": r"$\text{PU}^\text{it}$",
    "expected_entropies": r"$\text{AU}^\text{it}$",
    "expected_entropies_plus_expected_divergences": r"$\text{AU}^\text{it} + \text{EU}^\text{b}$",
    "one_minus_max_probs_of_fbar": r"$\max \bar{f}$",
    "one_minus_max_probs_of_bma": r"$\max \tilde{f}$",
    "one_minus_expected_max_probs": r"$\mathbb{E}\left[\max f\right]$",
    "expected_divergences": r"$\text{EU}^\text{b}$",
    "jensen_shannon_divergences": r"$\text{EU}^\text{it}$",
    "error_probabilities": r"$u^\text{cp}$",
    "duq_values": r"$u^\text{duq}$",
    "mahalanobis_values": r"$u^\text{mah}$",
    "risk_values": r"$u^\text{rp}$",
    "hard_bma_accuracy": None,
}


def main():
    wandb.login(key="341d19ab018aff60423f1ea0049fa41553ef94b4")
    api = wandb.Api()

    method_to_ids = {
        "GP": ("hx2ni3sr", "2vkuhe38"),
        "HET-XL": ("ktze6y0c", "3vnnnaix"),
        "Baseline": ("7y7e6kjf", "gypg5gc8"),
        "Dropout": ("f52l00hb", "9jztoaos"),
        "SNGP": ("us8v6277", "f32n7c05"),
        "Shallow Ens.": ("795iqrk8", "6r8nfwqc"),
        "Loss Pred.": ("kl7436jj", "960a6hfa"),
        "Corr. Pred.": ("iskn1vp6", "xsvl0zop"),
        "Deep Ens.": ("wzx8xxbn", "ymq2jv64"),
        "Laplace": ("tri75olb", "gkvfnbup"),
        "Mahalanobis": ("somhugzm", "swr2k8kf"),
    }

    create_directory("results")
    create_directory(f"results/correlation_matrix")

    metric_dict = {
        "log_prob_score_hard_bma_correctness": "Log Prob.",
        "brier_score_hard_bma_correctness": "Brier",
        "ece_hard_bma_correctness": "-ECE (*)",
        "auroc_hard_bma_correctness": "Correctness",
        "cumulative_hard_bma_abstinence_auc": "Abstinence",
        "hard_bma_accuracy": "Accuracy",
        "rank_correlation_bregman_au": "Aleatoric",
        "auroc_oodness": "OOD (*)",
    }

    performance_matrix_imagenet = np.zeros((len(metric_dict), len(method_to_ids), 3))
    performance_matrix_cifar = np.zeros((len(metric_dict), len(method_to_ids), 3))
    correlation_vector = np.zeros((len(metric_dict),))

    id_prefix = "best_id_test"
    mixture_prefix_cifar = "best_ood_test_soft/cifar10S2_mixed_soft/cifar10"
    mixture_prefix_imagenet = "best_ood_test_soft/imagenetS2_mixed_soft/imagenet"

    for k, distributional_estimator in enumerate(
        [
            "one_minus_max_probs_of_fbar",
            "one_minus_max_probs_of_bma",
            "one_minus_expected_max_probs",
        ]
    ):
        for j, (method_name, (imagenet_id, cifar_id)) in enumerate(
            tqdm(method_to_ids.items())
        ):
            sweep_imagenet = api.sweep(f"bmucsanyi/bias/{imagenet_id}")
            sweep_cifar = api.sweep(f"bmucsanyi/bias/{cifar_id}")

            for i, (metric_id, metric_name) in enumerate(metric_dict.items()):
                print(metric_id, metric_name)

                estimator_dict = {}

                for sweep, performance_matrix, anyam in zip(
                    [sweep_imagenet, sweep_cifar],
                    [performance_matrix_imagenet, performance_matrix_cifar],
                    [mixture_prefix_imagenet, mixture_prefix_cifar],
                ):
                    prefix = id_prefix if metric_name != "OOD (*)" else anyam
                    estimator_dict = {}
                    for run in sweep.runs:
                        for key in sorted(run.summary.keys()):
                            if key.startswith(prefix) and key.endswith(metric_id):
                                stripped_key = key.replace(f"{prefix}_", "").replace(
                                    f"_{metric_id}", ""
                                )

                                if (
                                    "mixed" in stripped_key
                                    or stripped_key not in ESTIMATOR_CONVERSION_DICT
                                ):
                                    continue

                                if stripped_key not in estimator_dict:
                                    estimator_dict[stripped_key] = [run.summary[key]]
                                else:
                                    estimator_dict[stripped_key].append(
                                        run.summary[key]
                                    )

                                if metric_name == "-ECE (*)":
                                    estimator_dict[stripped_key][-1] *= -1

                    for key in tuple(estimator_dict.keys()):
                        if "NaN" in estimator_dict[key]:
                            continue
                        estimator_dict[key] = np.mean(estimator_dict[key])

                    if len(estimator_dict) > 1:
                        if method_name == "Correctness Prediction":
                            estimator = "error_probabilities"
                        elif method_name == "DUQ":
                            estimator = "duq_values"
                        else:
                            estimator = distributional_estimator
                    else:
                        estimator = next(iter(estimator_dict.keys()))

                    performance_matrix[i, j, k] = estimator_dict[estimator]

    performance_matrix_imagenet = performance_matrix_imagenet.reshape(
        performance_matrix_imagenet.shape[0], -1
    )
    performance_matrix_cifar = performance_matrix_cifar.reshape(
        performance_matrix_cifar.shape[0], -1
    )

    for i in range(len(metric_dict)):
        perf_imagenet = performance_matrix_imagenet[i, :]
        perf_cifar = performance_matrix_cifar[i, :]
        correlation_vector[i] = spearmanr(perf_imagenet, perf_cifar)[0]

    print(correlation_vector)


if __name__ == "__main__":
    main()
