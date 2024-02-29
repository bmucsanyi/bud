import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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
    "gt_total_predictives_bregman_fbar": r"$\text{PU}^\text{b}$",
    "gt_biases_bregman_fbar": r"$\text{B}^\text{b}$",
    "gt_predictives_bregman_fbar": r"$\text{AU}^\text{b} + \text{B}^\text{b}$",
    "gt_aleatorics_bregman": r"$\text{AU}^\text{b}$",
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
        "gypg5gc8": "Baseline",
        "2vkuhe38": r"GP $\approx$ SNGP",
        "9jztoaos": "Dropout",
        "ymq2jv64": "Deep Ensemble",
    }

    dataset_conversion_dict = {
        "best_id_test": "CIFAR-10 Clean",
        "best_ood_test_soft/cifar10S1": "CIFAR-10 Severity 1",
        "best_ood_test_soft/cifar10S2": "CIFAR-10 Severity 2",
        "best_ood_test_soft/cifar10S3": "CIFAR-10 Severity 3",
        "best_ood_test_soft/cifar10S4": "CIFAR-10 Severity 4",
        "best_ood_test_soft/cifar10S5": "CIFAR-10 Severity 5",
    }

    create_directory("results")
    create_directory(f"results/ece_generalization")

    fig, ax = plt.subplots()
    axins = fig.add_axes([0.2, 0.5, 0.1, 0.4])

    for method_id, method_name in tqdm(id_to_method.items()):
        ece_list = []

        sweep = api.sweep(f"bmucsanyi/bias/{method_id}")
        suffix_ece = "ece_hard_bma_correctness"

        for prefix in dataset_conversion_dict:
            ece_dict = {}
            ece_list.append(ece_dict)
            for run in sweep.runs:
                for key in sorted(run.summary.keys()):
                    if key.startswith(prefix) and key.endswith(suffix_ece):
                        stripped_key = key.replace(f"{prefix}_", "").replace(
                            f"_{suffix_ece}", ""
                        )

                        if (
                            "mixed" in stripped_key
                            or stripped_key not in ESTIMATOR_CONVERSION_DICT
                        ):
                            continue

                        if stripped_key not in ece_dict:
                            ece_dict[stripped_key] = [run.summary[key]]
                        else:
                            ece_dict[stripped_key].append(run.summary[key])

        # Plotting the results for each method
        severities = np.arange(6)  # Severities from 0 to 5
        means = np.empty((6,))
        mins = np.empty((6,))
        maxs = np.empty((6,))

        for i, ece_dict in enumerate(ece_list):
            means_across_keys = {
                key: np.mean(ece_dict[key]) for key in ece_dict if "gt" not in key
            }
            best_estimator = min(means_across_keys.items(), key=lambda x: x[1])[0]

            values = ece_dict[best_estimator]
            means[i] = np.mean(values)
            mins[i] = np.min(values)
            maxs[i] = np.max(values)

        lower_errors = np.maximum(0, means - mins)
        upper_errors = np.maximum(0, maxs - means)
        error_bars = [
            lower_errors,
            upper_errors,
        ]

        ax.errorbar(
            severities,
            means,
            yerr=error_bars,
            fmt="-o",
            label=method_name,
            markersize=3,
            capsize=5,
        )

        axins.errorbar(
            severities,
            means,
            yerr=error_bars,
            fmt="-o",
            label=method_name,
            markersize=3,
            capsize=5,
        )

    axins.set_xlim(-0.1, 0.1)
    axins.set_ylim(0, 0.043)
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    mark_inset(ax, axins, loc1=2, loc2=2)

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("ECE Correctness")
    ax.set_ylim(0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [h[0] for h in handles],
        labels,
        bbox_to_anchor=(0.5, 1),
        loc="upper center",
        frameon=False,
    )
    ax.grid(True, linewidth=0.5)
    save_path = f"results/ece_generalization/ece.pdf"
    plt.savefig(save_path)
    plt.clf()  # Clear the figure for the next plot


if __name__ == "__main__":
    main()
