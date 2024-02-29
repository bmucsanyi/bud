import os
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.colors as mcolors

from tueplots import bundles

config = bundles.icml2022(family="serif", usetex=True, nrows=1, column="half")
config["figure.figsize"] = (3.25, 1.25)

plt.rcParams.update(config)

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
        "hx2ni3sr": r"GP $\approx$ SNGP",
        "f52l00hb": "Dropout",
        "wzx8xxbn": "Deep Ens.",
        "7y7e6kjf": "Baseline",
    }

    dataset_conversion_dict = {
        "best_id_test": "CIFAR-10 Clean",
        "best_ood_test_soft/imagenetS1": "CIFAR-10 Severity 1",
        "best_ood_test_soft/imagenetS2": "CIFAR-10 Severity 2",
        "best_ood_test_soft/imagenetS3": "CIFAR-10 Severity 3",
        "best_ood_test_soft/imagenetS4": "CIFAR-10 Severity 4",
        "best_ood_test_soft/imagenetS5": "CIFAR-10 Severity 5",
    }

    create_directory("results")
    create_directory(f"results/ece_generalization")

    _, ax = plt.subplots()

    bar_width = 0.15  # Width of the bars
    num_methods = len(id_to_method)

    def lighten_color(color, amount=0.5):
        """
        Lightens the given color by mixing it with white.
        :param color: Original color (hex or RGB).
        :param amount: Amount of white to mix in. 0 is the original color, 1 is white.
        :return: Lightened color in RGB format.
        """
        try:
            c = mcolors.to_rgb(color)  # Convert to RGB
        except ValueError:
            # If color is already RGB, use it as is
            c = color
        c_white = np.array([1, 1, 1])
        new_color = c + (c_white - c) * amount
        return new_color

    # Get the default color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Lighten each color
    lightened_colors = [
        lighten_color(color, amount=0.15) for color in default_colors
    ]  # Increase brightness by 15%

    for j, (method_id, method_name) in tqdm(enumerate(id_to_method.items())):
        ece_list = []

        sweep = api.sweep(f"bmucsanyi/bias/{method_id}")
        suffix_ece = "ece_hard_bma_correctness"

        for prefix in dataset_conversion_dict:
            ece_dict = {}
            ece_list.append(ece_dict)
            for run in sweep.runs:
                if run.state != "finished":
                    continue
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
            lower_errors[:2],
            upper_errors[:2],
        ]

        # Adjust the positions of the bars to group them by method
        bar_positions = (
            np.arange(2) - (num_methods * bar_width / 2) + j * bar_width + bar_width / 2
        )

        ax.bar(
            bar_positions,
            means[:2],
            bar_width,
            color=lightened_colors[j],
            label=method_name,
            zorder=2,
        )

        ax.errorbar(
            bar_positions,
            means[:2],
            yerr=error_bars,
            fmt="none",
            capsize=5,
            ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
            elinewidth=1,
            zorder=2,
        )

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(r"ECE $\downarrow$")
    ax.set_ylim([0, 0.15])
    ax.legend(frameon=False)

    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["ID", "OOD Severity 1"])
    ax.grid(axis="y", zorder=1, linewidth=0.5)
    save_path = f"results/ece_generalization/ece.pdf"
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":
    main()
