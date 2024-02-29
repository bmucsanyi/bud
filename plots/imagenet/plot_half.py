import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
import re
from matplotlib.patches import Patch

from tueplots import bundles

config = bundles.icml2022(family="serif", usetex=True, nrows=1, column="half")
config["figure.figsize"] = (3.25, 1.25)
# config["figure.figsize"] = (3.25, 1.4)

plt.rcParams.update(config)

from matplotlib.ticker import MultipleLocator

plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"

GT_LABELS = [
    r"$\text{PU}^\text{b}$",
    r"$\text{B}^\text{b}$",
    r"$\text{AU}^\text{b} + \text{B}^\text{b}$",
    r"$\text{AU}^\text{b}$",
]

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
    "hard_bma_accuracy": "",
}


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def plot_and_save(suffix, data, save_path, y_min, y_max):
    """Plots and saves the metric's bar chart with min-max error bars as a PDF."""
    means = [np.mean(values) for values in data.values()]
    mins = [np.min(values) for values in data.values()]
    maxs = [np.max(values) for values in data.values()]
    error_bars = [
        (mean - min_val, max_val - mean)
        for min_val, mean, max_val in zip(mins, means, maxs)
    ]
    labels = list(data.keys())

    # Combine and sort by means (decreasing order)
    combined = sorted(zip(labels, means, error_bars), key=lambda x: x[1], reverse=True)
    labels, means, error_bars = zip(*combined)

    _, ax = plt.subplots()
    ax.grid(axis="y", which="both", zorder=1)

    # Set major ticks at every 0.1 and minor ticks at every 0.05
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    bars = ax.bar(labels, means, yerr=np.array(error_bars).T, capsize=5, zorder=2)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(f"{suffix}")

    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)

    for bar, label in zip(bars, labels):
        lift = 0.2 if label == r"$\text{AU}^\text{b} + \text{B}^\text{b}$" else 0.03
        ax.text(
            bar.get_x() + bar.get_width() / 2 + 0.05,
            y_min + lift,
            label,
            ha="center",
            va="bottom",
            rotation="vertical",
            fontsize=6,
            zorder=3,
        )

        if label in GT_LABELS:
            bar.set_color(np.array([234.0, 67.0, 53.0]) / 255.0)
        else:
            bar.set_color(np.array([66.0, 103.0, 210.0]) / 255.0)

    # Add legend for bar colors
    legend_handles = [
        Patch(facecolor=np.array([66.0, 103.0, 210.0]) / 255.0, label="Estimate"),
        Patch(facecolor=np.array([234.0, 67.0, 53.0]) / 255.0, label="Ground Truth"),
    ]

    # Adding the legend with adjusted handlelength to make the patches more square-like
    ax.legend(
        frameon=False,
        handles=legend_handles,
        loc="upper right",
        fontsize="small",
        handlelength=1,
        ncol=2,
    )

    ax.set_ylim(bottom=y_min, top=y_max)
    plt.savefig(save_path)
    plt.close()


def plot_and_save_aggregated(
    suffix,
    best_metric,
    best_metric_mins_maxs,
    save_path,
    y_min,
    y_max,
    decreasing,
    label_offsets,
    offset_values,
):
    """Plots and saves the aggregated best values with min-max error bars as a PDF."""
    labels, best_values = zip(*best_metric.items())
    error_bars = [
        (best - min_val, max_val - best)
        for (min_val, max_val), best in zip(best_metric_mins_maxs.values(), best_values)
    ]

    # Combine and sort by best values
    combined = sorted(
        zip(labels, best_values, error_bars), key=lambda x: x[1], reverse=not decreasing
    )
    labels, best_values, error_bars = zip(*combined)

    _, ax = plt.subplots()
    ax.grid(axis="y", which="both", zorder=1, linewidth=0.5)
    # Set major ticks at every 0.1 and minor ticks at every 0.05
    multiplier = 2 if "Rank" in suffix else 1
    ax.yaxis.set_major_locator(MultipleLocator(0.1 * multiplier))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05 * multiplier))

    bars = ax.bar(
        labels,
        best_values,
        zorder=2,
    )
    ax.errorbar(
        labels,
        best_values,
        yerr=np.array(error_bars).T,
        fmt="none",
        ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
        elinewidth=1,
        capsize=5,
        zorder=3,
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(suffix + (r" $\uparrow$" if not decreasing else r" $\downarrow$"))

    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)

    label_offset_dict = dict(zip(label_offsets, offset_values))

    for bar, label in zip(bars, labels):
        if "$" in label:
            pattern = r"\$.*?\$"

            # Replace the matched pattern with an empty string
            processed_label = re.sub(pattern, "", label).strip()
        else:
            processed_label = label

        y_offset = label_offset_dict.get(
            processed_label, 0.03
        )  # Use the offset if available, otherwise default to 0.03
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_min + y_offset,  # Adjust the position using y_offset
            label,
            ha="center",
            va="bottom",
            rotation="vertical",
            zorder=4,
            fontsize=6,
        )

        if processed_label in POSTERIOR_ESTIMATORS:
            bar.set_color(np.array([52.0, 168.0, 83.0]) / 255.0)
        else:
            bar.set_color(np.array([251.0, 188.0, 4.0]) / 255.0)

        if processed_label == "Baseline":
            bar.set_color(np.array([154.0, 160.0, 166.0]) / 255.0)

    for i, (_, value) in enumerate(zip(bars, best_values)):
        ax.text(
            i,
            value + 1e-3,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=5,
            zorder=5,
        )

    # Add legend for bar colors
    legend_handles = [
        Patch(facecolor=np.array([52.0, 168.0, 83.0]) / 255.0, label="Distributional"),
        Patch(facecolor=np.array([251.0, 188.0, 4.0]) / 255.0, label="Deterministic"),
    ]

    # Adding the legend with adjusted handlelength to make the patches more square-like
    ax.legend(
        frameon=False,
        handles=legend_handles,
        loc="upper right",
        fontsize="small",
        handlelength=1,
        ncol=2,
    )

    ax.set_ylim(bottom=y_min, top=y_max)
    plt.savefig(save_path)
    plt.close()


def main(args):
    wandb.login(key="341d19ab018aff60423f1ea0049fa41553ef94b4")
    api = wandb.Api()

    id_to_method = {
        "hx2ni3sr": "GP",
        "ktze6y0c": "HET-XL",
        "7y7e6kjf": "Baseline",
        "f52l00hb": "Dropout",
        "us8v6277": "SNGP",
        "795iqrk8": "Shallow Ens.",
        "kl7436jj": "Loss Pred.",
        "iskn1vp6": "Corr. Pred.",
        "wzx8xxbn": "Deep Ens.",
        "tri75olb": "Laplace",
        "somhugzm": "Mahalanobis",
    }

    dataset_conversion_dict = {
        "best_id_test": "CIFAR-10 Clean",
        "best_ood_test_soft/imagenetS1": "CIFAR-10 Severity 1",
        "best_ood_test_soft/imagenetS2": "CIFAR-10 Severity 2",
        "best_ood_test_soft/imagenetS3": "CIFAR-10 Severity 3",
        "best_ood_test_soft/imagenetS4": "CIFAR-10 Severity 4",
        "best_ood_test_soft/imagenetS5": "CIFAR-10 Severity 5",
        "best_ood_test_soft/imagenetS1_mixed_soft/imagenet": "CIFAR-10 Clean + Severity 1",
        "best_ood_test_soft/imagenetS2_mixed_soft/imagenet": "CIFAR-10 Clean + Severity 2",
        "best_ood_test_soft/imagenetS3_mixed_soft/imagenet": "CIFAR-10 Clean + Severity 3",
        "best_ood_test_soft/imagenetS4_mixed_soft/imagenet": "CIFAR-10 Clean + Severity 4",
        "best_ood_test_soft/imagenetS5_mixed_soft/imagenet": "CIFAR-10 Clean + Severity 5",
    }

    if args.correct_auroc:

        def func(x):
            if isinstance(x, str):
                return x
            return max(x, 1 - x)

    elif args.correct_abs:

        def func(x):
            if isinstance(x, str):
                return x
            return abs(x)

    else:
        func = lambda x: x

    for prefix in dataset_conversion_dict:
        create_directory("results")
        create_directory(f"results/{args.metric}")
        create_directory(f"results/{args.metric}/{prefix.replace('/', '-')}")
        aggregated_estimators = {}
        aggregated_estimators_mins_maxs = {}

        for method_id, method_name in tqdm(id_to_method.items()):
            if args.only_posterior and method_name not in POSTERIOR_ESTIMATORS:
                continue
            sweep = api.sweep(f"bmucsanyi/bias/{method_id}")

            metric = {}
            suffix = args.metric

            for run in sweep.runs:
                if run.state != "finished":
                    continue
                for key in sorted(run.summary.keys()):
                    if key.startswith(prefix) and key.endswith(suffix):
                        stripped_key = key.replace(f"{prefix}_", "").replace(
                            f"_{suffix}", ""
                        )

                        if (
                            "mixed" in stripped_key
                            or stripped_key not in ESTIMATOR_CONVERSION_DICT
                        ):
                            continue

                        if ESTIMATOR_CONVERSION_DICT[stripped_key] not in metric:
                            metric[ESTIMATOR_CONVERSION_DICT[stripped_key]] = [
                                func(run.summary[key])
                            ]
                        else:
                            metric[ESTIMATOR_CONVERSION_DICT[stripped_key]].append(
                                func(run.summary[key])
                            )

            save_path = (
                f"results/{args.metric}/{prefix.replace('/', '-')}/{method_name}.pdf"
            )
            metric = {key: value for key, value in metric.items() if "NaN" not in value}

            if not metric:
                continue

            plot_and_save(
                args.title,
                metric,
                save_path,
                args.y_min,
                args.y_max,
            )

            if "accuracy" not in args.metric:
                if method_name == "Corr. Pred.":
                    aggregated_key = ESTIMATOR_CONVERSION_DICT["error_probabilities"]
                elif method_name == "Risk Pred.":
                    aggregated_key = ESTIMATOR_CONVERSION_DICT["risk_values"]
                elif method_name == "DUQ":
                    aggregated_key = ESTIMATOR_CONVERSION_DICT["duq_values"]
                elif method_name == "Mahalanobis":
                    aggregated_key = ESTIMATOR_CONVERSION_DICT["mahalanobis_values"]
                else:
                    aggregated_key = ESTIMATOR_CONVERSION_DICT.get(
                        args.distributional_estimator
                    )
            else:
                aggregated_key = None

            if aggregated_key is None:
                operator = min if args.decreasing else max
                means = {
                    key: np.mean(metric[key]) for key in metric if key not in GT_LABELS
                }
                aggregated_key = operator(means.items(), key=lambda x: x[1])[0]

            try:
                aggregated_estimators[method_name] = np.mean(metric[aggregated_key])
                aggregated_estimators_mins_maxs[method_name] = np.min(
                    metric[aggregated_key]
                ), np.max(metric[aggregated_key])
            except KeyError:
                continue

        if not aggregated_estimators:
            continue

        # Save the aggregated plot with min-max error bars
        aggregated_save_path = (
            f"results/{args.metric}/{prefix.replace('/', '-')}/aggregated.pdf"
        )
        plot_and_save_aggregated(
            args.title,
            aggregated_estimators,
            aggregated_estimators_mins_maxs,
            aggregated_save_path,
            args.y_min,
            args.y_max,
            args.decreasing,
            args.label_offsets,
            args.offset_values,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the metric for plotting.")
    parser.add_argument(
        "title", type=str, help="Name of the metric to be used in the analysis."
    )
    parser.add_argument(
        "metric", type=str, help="Name of the metric to be used in the analysis."
    )
    parser.add_argument("--distributional-estimator", type=str, default=None)
    parser.add_argument("--y-min", type=float, default=None, help="Minimum y value.")
    parser.add_argument("--y-max", type=float, default=None, help="Maximum y value.")
    parser.add_argument(
        "--label-offsets",
        nargs="*",
        default=[],
        help="List of labels that require y-offset adjustments.",
    )
    parser.add_argument(
        "--offset-values",
        nargs="*",
        type=float,
        default=[],
        help="List of y-offset values corresponding to the labels.",
    )
    parser.add_argument(
        "--decreasing",
        action="store_true",
        default=False,
        help="Whether the metric is increasing or decreasing.",
    )
    parser.add_argument(
        "--correct-auroc",
        action="store_true",
        default=False,
        help="no",
    )
    parser.add_argument(
        "--correct-abs",
        action="store_true",
        default=False,
        help="no",
    )
    parser.add_argument(
        "--only-posterior",
        action="store_true",
        default=False,
        help="no",
    )

    args = parser.parse_args()
    main(args)
