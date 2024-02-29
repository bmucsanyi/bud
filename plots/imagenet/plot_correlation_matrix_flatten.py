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

    id_to_method = {
        "hx2ni3sr": "GP",
        "ktze6y0c": "HET-XL",
        "7y7e6kjf": "Baseline",
        "f52l00hb": "Dropout",
        "us8v6277": "SNGP",
        "795iqrk8": "Shallow Ens.",
        "kl7436jj": "Risk Pred.",
        "iskn1vp6": "Corr. Pred.",
        "wzx8xxbn": "Deep Ens.",
        "o6wgl9no": "Deep Ens.+",
        "tri75olb": "Laplace",
        "somhugzm": "Mahalanobis",
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

    fig, ax = plt.subplots()

    performance_matrix = np.zeros((len(metric_dict), len(id_to_method), 3))
    correlation_matrix = np.zeros((len(metric_dict), len(metric_dict)))

    id_prefix = "best_id_test"
    mixture_prefix = "best_ood_test_soft/imagenetS2_mixed_soft/imagenet"

    for k, distributional_estimator in enumerate(
        [
            "one_minus_max_probs_of_fbar",
            "one_minus_max_probs_of_bma",
            "one_minus_expected_max_probs",
        ]
    ):
        for j, (method_id, method_name) in enumerate(tqdm(id_to_method.items())):
            sweep = api.sweep(f"bmucsanyi/bias/{method_id}")

            for i, (metric_id, metric_name) in enumerate(metric_dict.items()):
                prefix = id_prefix if metric_name != "OOD (*)" else mixture_prefix

                estimator_dict = {}

                for run in sweep.runs:
                    if run.state != "finished": continue
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
                                estimator_dict[stripped_key].append(run.summary[key])

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

    performance_matrix = performance_matrix.reshape(performance_matrix.shape[0], -1)

    for i in range(len(metric_dict)):
        for j in range(len(metric_dict)):
            perf_i = performance_matrix[i, :]
            perf_j = performance_matrix[j, :]
            correlation_matrix[i, j] = spearmanr(perf_i, perf_j)[0]

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=0)

    # Choose a diverging colormap
    cmap = plt.get_cmap("coolwarm")

    # Plot the heatmap, applying the mask
    cax = ax.imshow(
        np.ma.masked_where(mask, correlation_matrix),
        interpolation="nearest",
        cmap=cmap,
        vmin=-1,
        vmax=1,  # Set the scale of the colormap from -1 to 1
    )

    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(width=0.1)
    cbar.set_ticks([-0.987, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"])

    # Set ticks
    ax.set_xticks(np.arange(len(metric_dict)))
    ax.set_yticks(np.arange(len(metric_dict)))

    # Set tick labels
    ax.set_xticklabels(metric_dict.values())
    ax.set_yticklabels(metric_dict.values())

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations for only the lower triangle
    for i in range(len(metric_dict)):
        for j in range(len(metric_dict)):
            if i > j:
                ax.text(
                    j,
                    i,
                    round(correlation_matrix[i, j], 3),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=5,
                )
    ax.spines[["right", "top"]].set_visible(False)
    plt.savefig("results/correlation_matrix/correlation_matrix.pdf")


if __name__ == "__main__":
    main()
