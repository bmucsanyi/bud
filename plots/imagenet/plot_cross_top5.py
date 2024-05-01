import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
import sys
import json

sys.path.insert(0, "..")

from utils import ID_TO_METHOD_IMAGENET, create_directory

from tueplots import bundles

plt.rcParams.update(
    bundles.icml2022(family="serif", usetex=True, nrows=1, column="half")
)

plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"


def main():
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    with open("../../wandb_key.json") as f:
        wandb_key = json.load(f)["key"]

    wandb.login(key=wandb_key)
    api = wandb.Api()

    dataset_list = [
        ("best_id_test", None),
        (
            "best_ood_test_soft/imagenetS1",
            "best_ood_test_soft/imagenetS1_mixed_soft/imagenet",
        ),
        (
            "best_ood_test_soft/imagenetS2",
            "best_ood_test_soft/imagenetS2_mixed_soft/imagenet",
        ),
        (
            "best_ood_test_soft/imagenetS3",
            "best_ood_test_soft/imagenetS3_mixed_soft/imagenet",
        ),
        (
            "best_ood_test_soft/imagenetS4",
            "best_ood_test_soft/imagenetS4_mixed_soft/imagenet",
        ),
        (
            "best_ood_test_soft/imagenetS5",
            "best_ood_test_soft/imagenetS5_mixed_soft/imagenet",
        ),
    ]

    create_directory("results")
    create_directory("results/corr_vs_acc_top5")

    for method_id, method_name in tqdm(ID_TO_METHOD_IMAGENET.items()):
        sweep = api.sweep(f"bmucsanyi/bias/{method_id}")
        suffix_correctness = "auroc_hard_bma_correctness_top5"
        suffix_abstinence = "cumulative_hard_bma_abstinence_auc_top5"

        num_successful_runs = sum(1 for run in sweep.runs if run.state == "finished")
        auroc_correctness_matrix = np.zeros(
            (num_successful_runs, 6)
        )  # [runs, severities]
        auc_abstinence_matrix = np.zeros((num_successful_runs, 6))  # [runs, severities]
        accuracy_matrix = np.zeros((num_successful_runs, 6))

        for j, (prefix_normal, _) in enumerate(dataset_list):
            if method_name == "Corr. Pred.":
                key_auroc = "_error_probabilities_"
            elif method_name == "Loss Pred.":
                key_auroc = "_risk_values_"
            elif method_name == "Mahalanobis":
                key_auroc = "_mahalanobis_values_"
            else:
                key_auroc = "_one_minus_max_probs_of_bma_"

            key_accuracy = "_hard_bma_accuracy_top5"

            i = 0
            for run in sweep.runs:
                if run.state != "finished":
                    continue
                auroc_correctness_matrix[i, j] = run.summary[
                    prefix_normal + key_auroc + suffix_correctness
                ]
                auc_abstinence_matrix[i, j] = run.summary[
                    prefix_normal + key_auroc + suffix_abstinence
                ]
                accuracy_matrix[i, j] = run.summary[prefix_normal + key_accuracy]
                i += 1

        save_path = f"results/corr_vs_acc_top5/{method_name.replace('.', '').replace(' ', '_')}.pdf"

        # Plotting the results for each method
        severities = np.arange(6)  # Severities from 0 to 5
        means_auroc_correctness = np.mean(auroc_correctness_matrix, axis=0)
        min_values_auroc_correctness = np.min(auroc_correctness_matrix, axis=0)
        max_values_auroc_correctness = np.max(auroc_correctness_matrix, axis=0)

        means_auc_abstinence = np.mean(auc_abstinence_matrix, axis=0)
        min_values_auc_abstinence = np.min(auc_abstinence_matrix, axis=0)
        max_values_auc_abstinence = np.max(auc_abstinence_matrix, axis=0)

        means_accuracy = np.mean(accuracy_matrix, axis=0)
        min_values_accuracy = np.min(accuracy_matrix, axis=0)
        max_values_accuracy = np.max(accuracy_matrix, axis=0)

        # Calculate the differences for the error bars
        lower_errors_auroc_correctness = np.maximum(
            0, means_auroc_correctness - min_values_auroc_correctness
        )
        upper_errors_auroc_correctness = np.maximum(
            0, max_values_auroc_correctness - means_auroc_correctness
        )
        error_bars_auroc_correctness = [
            lower_errors_auroc_correctness,
            upper_errors_auroc_correctness,
        ]

        lower_errors_auc_abstinence = np.maximum(
            0, means_auc_abstinence - min_values_auc_abstinence
        )
        upper_errors_auc_abstinence = np.maximum(
            0, max_values_auc_abstinence - means_auc_abstinence
        )
        error_bars_auc_abstinence = [
            lower_errors_auc_abstinence,
            upper_errors_auc_abstinence,
        ]

        # Calculate the differences for the error bars
        lower_errors_accuracy = np.maximum(0, means_accuracy - min_values_accuracy)
        upper_errors_accuracy = np.maximum(0, max_values_accuracy - means_accuracy)
        error_bars_accuracy = [lower_errors_accuracy, upper_errors_accuracy]

        plt.gca().spines[["right", "top"]].set_visible(False)
        plt.errorbar(
            severities,
            2 * (means_auroc_correctness - 0.5),
            yerr=2 * np.array(error_bars_auroc_correctness),
            fmt="-o",
            label="AUROC Correctness",
            color=default_colors[0],
            markersize=3,
            ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
            elinewidth=1,
            capsize=5,
        )
        plt.errorbar(
            severities,
            means_auroc_correctness,
            yerr=error_bars_auroc_correctness,
            fmt="--o",
            color=default_colors[0],
            markersize=3,
            ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
            elinewidth=1,
            capsize=5,
            alpha=0.2,
        )
        plt.errorbar(
            severities,
            (1 - means_accuracy) ** (-1) * (means_auc_abstinence - means_accuracy),
            yerr=(1 - means_accuracy) ** (-1) * np.array(error_bars_auc_abstinence),
            fmt="-o",
            label="AUC Abstinence",
            color=default_colors[1],
            markersize=3,
            ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
            elinewidth=1,
            capsize=5,
        )
        plt.errorbar(
            severities,
            means_auc_abstinence,
            yerr=error_bars_auc_abstinence,
            fmt="--o",
            color=default_colors[1],
            markersize=3,
            ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
            elinewidth=1,
            capsize=5,
            alpha=0.2,
        )
        plt.errorbar(
            severities,
            1000 / 999 * (means_accuracy - 0.001),
            yerr=1000 / 999 * np.array(error_bars_accuracy),
            fmt="-o",
            label="Accuracy",
            color=default_colors[2],
            markersize=3,
            ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
            elinewidth=1,
            capsize=5,
        )
        plt.errorbar(
            severities,
            means_accuracy,
            yerr=error_bars_accuracy,
            fmt="--o",
            color=default_colors[2],
            markersize=3,
            ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
            elinewidth=1,
            capsize=5,
            alpha=0.2,
        )

        plt.xlabel("Severity Level")
        plt.ylabel(rf"Metric Values $\uparrow$")
        plt.ylim(0, 1)

        handles, labels = plt.gca().get_legend_handles_labels()
        handles = [h[0] for h in handles]

        plt.legend(handles, labels, frameon=False)
        plt.grid(True, linewidth=0.5)
        plt.savefig(save_path)
        plt.clf()  # Clear the figure for the next plot


if __name__ == "__main__":
    main()
