import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import wandb

from tueplots import bundles
from scipy.stats import spearmanr
import sys
import json

sys.path.insert(0, "..")

from utils import (
    ESTIMATOR_CONVERSION_DICT,
    create_directory,
)

plt.rcParams.update(bundles.icml2024(family="serif", column="half", usetex=True))

plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"


def main():
    # Add accuracy to estimator dict
    ESTIMATOR_CONVERSION_DICT["hard_bma_accuracy"] = "none"

    with open("../../wandb_key.json") as f:
        wandb_key = json.load(f)["key"]

    wandb.login(key=wandb_key)
    api = wandb.Api()

    create_directory("results")
    create_directory("results/correlation_matrix")

    metric_dict = {
        "auroc_hard_bma_correctness": "Correctness AUROC",
        "ece_hard_bma_correctness": "-ECE (*)",
        "brier_score_hard_bma_correctness": "Brier Score",
        "log_prob_score_hard_bma_correctness": "Log Prob. Score",
        "cumulative_hard_bma_abstinence_auc": "Abstinence AUC",
        "hard_bma_accuracy": "Accuracy",
        "rank_correlation_bregman_au": "Aleatoric Rank Corr.",
        "auroc_oodness": "OOD AUROC (*)",
    }

    id_to_method = {
        "46elax73": "GP",
        "ktze6y0c": "HET-XL",
        "3zt619eq": "CE Baseline",
        "f52l00hb": "MC-Dropout",
        "ew6b0m1x": "SNGP",
        "795iqrk8": "Shallow Ens.",
        "iskn1vp6": "Corr. Pred.",
        "1nz1l6qj": "Deep Ens.",
        "0qpln50b": "Laplace",
        "yxvvtw51": "Temperature",
        "5exmovzc": "DDU",
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
                prefix = id_prefix if metric_name != "OOD AUROC (*)" else mixture_prefix

                estimator_dict = {}

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
                                or stripped_key not in ESTIMATOR_CONVERSION_DICT
                                or "gt" in stripped_key
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
                    if method_name == "Corr. Pred.":
                        estimator = "error_probabilities"
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

    # Choose a diverging colormap
    cmap = plt.get_cmap("coolwarm")

    # Plot the heatmap, applying the mask
    cax = ax.imshow(
        correlation_matrix,
        interpolation="nearest",
        cmap=cmap,
        vmin=-1,
        vmax=1,  # Set the scale of the colormap from -1 to 1
    )

    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(width=0.1)
    cbar.set_ticks([-0.983, 0, 1.01])
    # cbar.set_ticklabels(["-1 (Neg. Corr.)", "0 (No Corr.)", "1 (Pos. Corr.)"])
    cbar.set_ticklabels(["-1", "0", "1"])

    # Set ticks
    ax.set_xticks(np.arange(len(metric_dict)))
    ax.set_yticks(np.arange(len(metric_dict)))

    # Set tick labels
    ax.set_xticklabels(metric_dict.values())
    ax.set_yticklabels(metric_dict.values())

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations for only the lower triangle
    for i in range(len(metric_dict)):
        for j in range(len(metric_dict)):
            ax.text(
                j,
                i,
                round(correlation_matrix[i, j], 2),
                ha="center",
                va="center",
                color="black",
                fontsize=5,
            )
    ax.spines[["right", "top"]].set_visible(False)
    plt.savefig("results/correlation_matrix/correlation_matrix.pdf")


if __name__ == "__main__":
    main()
