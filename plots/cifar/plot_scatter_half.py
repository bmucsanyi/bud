import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr

from tueplots import bundles

plt.rcParams.update(
    bundles.icml2022(family="serif", usetex=True, nrows=1, column="half")
)
from tueplots.constants.color import rgb

plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"

GT_LABELS = [
    r"$\text{PU}^\text{b}$",
    r"$\text{B}^\text{b}$",
    r"$\text{AU}^\text{b} + \text{B}^\text{b}$",
    r"$\text{AU}^\text{b}$",
]

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

POSTERIOR_ESTIMATORS = [
    "GP",
    "HET-XL",
    "Baseline",
    "Dropout",
    "SNGP",
    "Shallow Ens.",
    "Deep Ens.",
    "Laplace",
]


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def plot_and_save(label_x, label_y, data_x, data_y, save_path):
    """Plots and saves the metric's bar chart with min-max error bars as a PDF."""
    _, ax = plt.subplots()
    ax.grid(zorder=1)
    ax.scatter(data_x, data_y, zorder=2, color=rgb.tue_green)
    ax.spines[["right", "top"]].set_visible(False)

    ax.set_xlabel(f"{label_x}")
    ax.set_ylabel(f"{label_y}")

    plt.savefig(save_path)
    plt.close()


def plot_and_save_aggregated(
    label_x,
    label_y,
    data_x,
    data_y,
    save_path,
):
    """Plots and saves the aggregated best values with min-max error bars as a PDF."""
    full_x = []
    full_y = []

    fig, ax = plt.subplots()
    ax.grid(zorder=1)
    for key in data_x:
        values_x = data_x[key]
        values_y = data_y[key]
        ax.scatter(values_x, values_y, zorder=2, label=key, s=10)

        full_x.extend(values_x)
        full_y.extend(values_y)

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel(f"{label_x}")
    ax.set_ylabel(f"{label_y} (rcorr: {spearmanr(full_x, full_y)[0]:.4f})")
    fig.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    plt.savefig(save_path)
    plt.close()


def main(args):
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
        "960a6hfa": "Loss Pred.",
        "xsvl0zop": "Corr. Pred.",
        "ymq2jv64": "Deep Ens.",
        "gkvfnbup": "Laplace",
        "swr2k8kf": "Mahalanobis",
    }

    dataset_conversion_dict = {
        "best_id_test": "CIFAR-10 Clean",
        "best_ood_test_soft/cifar10S1": "CIFAR-10 Severity 1",
        "best_ood_test_soft/cifar10S2": "CIFAR-10 Severity 2",
        "best_ood_test_soft/cifar10S3": "CIFAR-10 Severity 3",
        "best_ood_test_soft/cifar10S4": "CIFAR-10 Severity 4",
        "best_ood_test_soft/cifar10S5": "CIFAR-10 Severity 5",
        "best_ood_test_soft/cifar10S1_mixed_soft/cifar10": "CIFAR-10 Clean + Severity 1",
        "best_ood_test_soft/cifar10S2_mixed_soft/cifar10": "CIFAR-10 Clean + Severity 2",
        "best_ood_test_soft/cifar10S3_mixed_soft/cifar10": "CIFAR-10 Clean + Severity 3",
        "best_ood_test_soft/cifar10S4_mixed_soft/cifar10": "CIFAR-10 Clean + Severity 4",
        "best_ood_test_soft/cifar10S5_mixed_soft/cifar10": "CIFAR-10 Clean + Severity 5",
    }

    suffix_x = args.metric_x
    suffix_y = args.metric_y

    for prefix in dataset_conversion_dict:
        create_directory("results")
        create_directory(f"results/{args.metric_x}_{args.metric_y}")
        create_directory(
            f"results/{args.metric_x}_{args.metric_y}/{prefix.replace('/', '-')}"
        )
        data_xs = {}
        data_ys = {}

        for method_id, method_name in tqdm(id_to_method.items()):
            sweep = api.sweep(f"bmucsanyi/bias/{method_id}")

            metric_x = {}
            metric_y = {}

            for run in sweep.runs:
                for key in sorted(run.summary.keys()):
                    for suffix, metric in zip(
                        [suffix_x, suffix_y], [metric_x, metric_y]
                    ):
                        if key.startswith(prefix) and key.endswith(suffix):
                            stripped_key = key.replace(f"{prefix}_", "").replace(
                                f"_{suffix}", ""
                            )

                            if (
                                "mixed" in stripped_key
                                or stripped_key not in ESTIMATOR_CONVERSION_DICT
                            ):
                                continue

                            if stripped_key not in metric:
                                metric[stripped_key] = [run.summary[key]]
                            else:
                                metric[stripped_key].append(run.summary[key])

            save_path = (
                f"results/{args.metric_x}_{args.metric_y}/{prefix.replace('/', '-')}/"
                f"{method_name.replace('.', '').replace(' ', '_')}.pdf"
            )
            metric_x = {
                key: value for key, value in metric_x.items() if "NaN" not in value
            }
            metric_y = {
                key: value for key, value in metric_y.items() if "NaN" not in value
            }

            if not metric_x or not metric_y:
                continue

            if len(metric_x.keys()) > 1:
                if method_name == "Corr. Pred.":
                    key_x = "error_probabilities"
                elif method_name == "Loss Pred.":
                    key_x = "risk_values"
                elif method_name == "DUQ":
                    key_x = "duq_values"
                elif method_name == "Mahalanobis":
                    key_x = "mahalanobis_values"
                else:
                    key_x = args.distributional_estimator
            else:
                key_x = list(metric_x.keys())[0]  # e.g. accuracy

            if key_x is None and not args.optimize_y:
                operator = min if args.decreasing else max
                means_x = {
                    key: np.mean(metric_x[key])
                    for key in metric_x
                    if "gt" not in key and key in metric_y
                }
                key_x = operator(means_x.items(), key=lambda x: x[1])[0]

            if len(metric_y.keys()) > 1:
                if method_name == "Corr. Pred.":
                    key_y = "error_probabilities"
                elif method_name == "Loss Pred.":
                    key_y = "risk_values"
                elif method_name == "DUQ":
                    key_y = "duq_values"
                elif method_name == "Mahalanobis":
                    key_y = "mahalanobis_values"
                else:
                    key_y = key_x
            else:
                key_y = list(metric_y.keys())[0]

            if key_y is None and args.optimize_y:
                operator = min if args.decreasing else max
                means_y = {
                    key: np.mean(metric_y[key])
                    for key in metric_y
                    if "gt" not in key and key in metric_x
                }
                key_y = operator(means_y.items(), key=lambda x: x[1])[0]
                key_x = key_y

            if key_x not in metric_x or key_y not in metric_y:
                continue

            data_xs[method_name] = metric_x[key_x]
            data_ys[method_name] = metric_y[key_y]

            plot_and_save(
                args.label_x,
                args.label_y,
                metric_x[key_x],
                metric_y[key_y],
                save_path,
            )

        if not data_xs or not data_ys:
            continue

        # Save the aggregated plot with min-max error bars
        aggregated_save_path = f"results/{args.metric_x}_{args.metric_y}/{prefix.replace('/', '-')}/aggregated.pdf"
        plot_and_save_aggregated(
            args.label_x, args.label_y, data_xs, data_ys, aggregated_save_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the metric for plotting.")
    parser.add_argument("label_x", type=str, help="Label of x axis.")
    parser.add_argument("label_y", type=str, help="Label of y axis.")
    parser.add_argument("metric_x", type=str, help="Name of the x-axis metric.")
    parser.add_argument("metric_y", type=str, help="Name of the y-axis metric.")
    parser.add_argument("--distributional-estimator", type=str, default=None)
    parser.add_argument(
        "--optimize-y",
        action="store_true",
        default=False,
        help="Whether the metric is increasing or decreasing.",
    )
    parser.add_argument(
        "--decreasing",
        action="store_true",
        default=False,
        help="Whether the metric is increasing or decreasing.",
    )

    args = parser.parse_args()
    main(args)
