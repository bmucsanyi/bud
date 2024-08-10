import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import wandb

from tueplots import bundles
from scipy.stats import pearsonr, spearmanr
import sys
import json

sys.path.insert(0, "..")

from utils import (
    ESTIMATOR_CONVERSION_DICT,
    create_directory,
)

# plt.rcParams.update(bundles.icml2024(family="serif", column="half", usetex=True))
plt.rcParams.update(bundles.neurips2024())
plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"


def main():
    # Add to estimator dict
    ESTIMATOR_CONVERSION_DICT["hard_bma_accuracy_original"] = "none"
    ESTIMATOR_CONVERSION_DICT["log_prob_score_hard_bma_aleatoric_original"] = "none"
    ESTIMATOR_CONVERSION_DICT["brier_score_hard_fbar_aleatoric_original"] = "none"

    with open("../../wandb_key.json") as f:
        wandb_key = json.load(f)["key"]

    wandb.login(key=wandb_key)
    api = wandb.Api()

    create_directory("results")
    create_directory("results/correlation_matrix")

    metric_dict = {
        "auroc_hard_bma_correctness_original": "Correctness AUROC",
        "ece_hard_bma_correctness_original": "-ECE",
        "brier_score_hard_bma_correctness_original": "Correctness Brier",
        "log_prob_score_hard_bma_correctness_original": "Correctness Log Prob.",
        "hard_bma_raulc_original": "rAULC",
        "hard_bma_eaurc_original": "-E-AURC",
        "cumulative_hard_bma_abstinence_auc_original": "AUAC",
        "hard_bma_accuracy_original": "Accuracy",
        "log_prob_score_hard_bma_aleatoric_original": "Aleatoric Log Prob.",
        "brier_score_hard_fbar_aleatoric_original": "Aleatoric Brier",
        "rank_correlation_bregman_au": "Aleatoric Rank Corr.",
        "auroc_multiple_labels": "Aleatoric AUROC",
        "auroc_oodness": "OOD AUROC",
    }

    id_to_method = {
        "0zh85pjp": "GP",
        "n6ocb8vt": "HET-XL",
        "75316qay": "CE Baseline",
        "iphs7vdj": "MC-Dropout",
        "5l11sz1l": "SNGP",
        "50dvkkny": "Shallow Ens.",
        "qthh97bn": "Loss Pred.",
        "7bexzi5z": "Corr. Pred.",
        "oyn8zlw5": "Deep Ens.",
        "i170wvxa": "Laplace",
        "iovcgd69": "Mahalanobis",
        "9mqh7if3": "Temperature",
        "n5g7bnct": "DDU",
        "7yusrr4s": "HET",
    }

    performance_matrix = np.zeros((len(metric_dict), len(id_to_method), 3))
    correlation_matrix_spearman = np.zeros((len(metric_dict), len(metric_dict)))
    correlation_matrix_pearson = np.zeros((len(metric_dict), len(metric_dict)))

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
                prefix = id_prefix if metric_name != "OOD AUROC" else mixture_prefix

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

                            if metric_name in ["-ECE", "-E-AURC"]:
                                estimator_dict[stripped_key][-1] *= -1

                for key in tuple(estimator_dict.keys()):
                    if "NaN" in estimator_dict[key]:
                        continue
                    estimator_dict[key] = np.mean(estimator_dict[key])

                if "brier_score_hard_fbar_aleatoric_original" in estimator_dict:
                    estimator_dict = {
                        "brier_score_hard_fbar_aleatoric_original": estimator_dict[
                            "brier_score_hard_fbar_aleatoric_original"
                        ]
                    }

                if len(estimator_dict) > 1:
                    if method_name == "Corr. Pred.":
                        estimator = "error_probabilities"
                    else:
                        estimator = distributional_estimator
                else:
                    # print(estimator_dict)
                    estimator = next(iter(estimator_dict.keys()))

                # print(list(estimator_dict.keys()))
                performance_matrix[i, j, k] = estimator_dict[estimator]

    performance_matrix = performance_matrix.reshape(performance_matrix.shape[0], -1)

    for i in range(len(metric_dict)):
        for j in range(len(metric_dict)):
            perf_i = performance_matrix[i, :]
            perf_j = performance_matrix[j, :]
            correlation_matrix_spearman[i, j] = spearmanr(perf_i, perf_j)[0]
            correlation_matrix_pearson[i, j] = pearsonr(perf_i, perf_j)[0]

    correlation_matrices = [correlation_matrix_spearman, correlation_matrix_pearson]
    names = ["correlation_matrix_spearman", "correlation_matrix_pearson"]

    for correlation_matrix, name in zip(correlation_matrices, names):
        fig, ax = plt.subplots()
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
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
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
        plt.savefig(f"results/correlation_matrix/{name}.pdf")


if __name__ == "__main__":
    main()
