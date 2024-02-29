import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb

from tueplots import bundles

plt.rcParams.update(
    bundles.icml2022(family="serif", usetex=True, nrows=1, column="half")
)

plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    # Get the default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    wandb.login(key="341d19ab018aff60423f1ea0049fa41553ef94b4")
    api = wandb.Api()

    id_to_method = {
        "2vkuhe38": "GP",
        "3vnnnaix": "HET-XL",
        "gypg5gc8": "Baseline",
        "9jztoaos": "Dropout",
        "f32n7c05": "SNGP",
        "03coev3u": "DUQ",
        "6r8nfwqc": "Shallow Ens.",
        "960a6hfa": "Risk Pred.",
        "xsvl0zop": "Corr. Pred.",
        "ymq2jv64": "Deep Ens.",
        "gkvfnbup": "Laplace",
        "swr2k8kf": "Mahalanobis",
    }

    dataset_list = [
        ("best_id_test", None),
        (
            "best_ood_test_soft/cifar10S1",
            "best_ood_test_soft/cifar10S1_mixed_soft/cifar10",
        ),
        (
            "best_ood_test_soft/cifar10S2",
            "best_ood_test_soft/cifar10S2_mixed_soft/cifar10",
        ),
        (
            "best_ood_test_soft/cifar10S3",
            "best_ood_test_soft/cifar10S3_mixed_soft/cifar10",
        ),
        (
            "best_ood_test_soft/cifar10S4",
            "best_ood_test_soft/cifar10S4_mixed_soft/cifar10",
        ),
        (
            "best_ood_test_soft/cifar10S5",
            "best_ood_test_soft/cifar10S5_mixed_soft/cifar10",
        ),
    ]

    create_directory("results")
    create_directory(f"results/corr_vs_acc")

    for method_id, method_name in tqdm(id_to_method.items()):
        auroc_correctness_matrix = np.zeros((5, 6))  # [runs, severities]
        auroc_oodness_matrix = np.zeros((5, 6))  # [runs, severities]
        auc_abstinence_matrix = np.zeros((5, 6))  # [runs, severities]
        accuracy_matrix = np.zeros((5, 6))

        sweep = api.sweep(f"bmucsanyi/bias/{method_id}")
        suffix_correctness = "auroc_hard_bma_correctness"
        suffix_oodness = "auroc_oodness"
        suffix_abstinence = "cumulative_hard_bma_abstinence_auc"

        for j, (prefix_normal, prefix_oodness) in enumerate(dataset_list):
            if method_name == "Correctness Prediction":
                key_auroc = "_error_probabilities_"
            elif method_name == "Risk Prediction":
                key_auroc = "_risk_values_"
            elif method_name == "DUQ":
                key_auroc = "_duq_values_"
            elif method_name == "Mahalanobis":
                key_auroc = "_mahalanobis_values_"
            else:
                key_auroc = "_one_minus_max_probs_of_fbar_"
            key_accuracy = "_hard_bma_accuracy"

            for i, run in enumerate(sweep.runs):
                auroc_correctness_matrix[i, j] = run.summary[
                    prefix_normal + key_auroc + suffix_correctness
                ]
                auc_abstinence_matrix[i, j] = run.summary[
                    prefix_normal + key_auroc + suffix_abstinence
                ]

                if prefix_oodness is None:
                    auroc_oodness_matrix[i, j] = np.nan
                else:
                    auroc_oodness_matrix[i, j] = run.summary[
                        prefix_oodness + key_auroc + suffix_oodness
                    ]
                accuracy_matrix[i, j] = run.summary[prefix_normal + key_accuracy]

            if i < 4:
                auroc_correctness_matrix[:, j] = auroc_correctness_matrix[i, j]
                auc_abstinence_matrix[:, j] = auc_abstinence_matrix[i, j]
                auroc_oodness_matrix[:, j] = auroc_oodness_matrix[i, j]
                accuracy_matrix[:, j] = accuracy_matrix[i, j]

        save_path = f"results/corr_vs_acc/{method_name}.pdf"

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
            yerr=error_bars_auroc_correctness,
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
            alpha=0.2
        )
        plt.errorbar(
            severities,
            (1 - means_accuracy)**(-1) * (means_auc_abstinence - means_accuracy),
            yerr=error_bars_auc_abstinence,
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
            alpha=0.2
        )
        plt.errorbar(
            severities,
            10/9 * (means_accuracy - 0.1),
            yerr=error_bars_accuracy,
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
            alpha=0.2
        )

        plt.xlabel("Severity Level")
        plt.ylabel(rf"AUROCs and Acc. $\uparrow$")
        plt.ylim(0, 1)

        handles, labels = plt.gca().get_legend_handles_labels()
        handles = [h[0] for h in handles]

        plt.legend(handles, labels, frameon=False)
        plt.grid(True, linewidth=0.5)
        plt.savefig(save_path)
        plt.clf()  # Clear the figure for the next plot


if __name__ == "__main__":
    main()
