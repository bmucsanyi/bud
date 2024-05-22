import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
import re
from matplotlib.patches import Patch
import sys
import json

sys.path.insert(0, "..")

from utils import (
    POSTERIOR_ESTIMATORS,
    GT_LABELS,
    ID_TO_METHOD_CIFAR,
    DATASET_CONVERSION_DICT_CIFAR,
    ESTIMATOR_CONVERSION_DICT,
    create_directory,
)

from tueplots import bundles

config = bundles.neurips2023(family="serif", usetex=True, nrows=1, ncols=1)
config["figure.figsize"] = (2.75, 1.1)

plt.rcParams.update(config)

from matplotlib.ticker import MultipleLocator

plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"


def plot_and_save(suffix, data, save_path):
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
        ax.text(
            bar.get_x() + bar.get_width() / 2 + 0.05,
            0.5 + 0.03,
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

    # Add the mean values on top of the bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1e-3 + error_bars[i][1],
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=5,
            zorder=4,
        )

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

    ax.set_ylim(bottom=0.5, top=1)
    plt.savefig(save_path)
    plt.close()


def plot_and_save_aggregated(
    suffix,
    best_metric,
    best_metric_mins_maxs,
    save_path,
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
        zip(labels, best_values, error_bars), key=lambda x: x[1], reverse=True
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
    ax.set_ylabel(suffix + r" $\uparrow$")

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
            0.5 + y_offset,  # Adjust the position using y_offset
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

        if processed_label == "CE Baseline":
            bar.set_color(np.array([154.0, 160.0, 166.0]) / 255.0)

    # for i, (_, value) in enumerate(zip(bars, best_values)):
    #     ax.text(
    #         i,
    #         value + 1e-3,
    #         f"{value:.3f}",
    #         ha="center",
    #         va="bottom",
    #         fontsize=5,
    #         zorder=5,
    #     )

    # Add legend for bar colors
    legend_handles = [
        Patch(facecolor=np.array([52.0, 168.0, 83.0]) / 255.0, label="Distributional"),
        Patch(facecolor=np.array([251.0, 188.0, 4.0]) / 255.0, label="Deterministic"),
    ]

    ax.legend(
        frameon=False,
        handles=legend_handles,
        loc="upper right",
        fontsize="small",
        handlelength=1,
    )

    ax.set_ylim(bottom=0.5, top=1)
    plt.savefig(save_path)
    plt.close()


def main(args):
    with open("../../wandb_key.json") as f:
        wandb_key = json.load(f)["key"]

    wandb.login(key=wandb_key)
    api = wandb.Api()

    def func(x):
        if isinstance(x, str):
            return x
        return max(x, 1 - x)

    for prefix in DATASET_CONVERSION_DICT_CIFAR:
        create_directory("results")
        create_directory(f"results/gp_eu")
        create_directory(f"results/gp_eu/{prefix.replace('/', '-')}")
        aggregated_estimators = {}
        aggregated_estimators_mins_maxs = {}

        for method_id, method_name in tqdm(ID_TO_METHOD_CIFAR.items()):
            sweep = api.sweep(f"bmucsanyi/bias/{method_id}")

            metric = {}
            suffix = "auroc_oodness"

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
                f"results/gp_eu/{prefix.replace('/', '-')}/"
                f"{method_name.replace('.', '').replace(' ', '_')}.pdf"
            )
            metric = {key: value for key, value in metric.items() if "NaN" not in value}

            if not metric:
                continue

            plot_and_save(
                "AUROC",
                metric,
                save_path,
            )

            if method_name == "Corr. Pred.":
                aggregated_key = ESTIMATOR_CONVERSION_DICT["error_probabilities"]
            elif method_name == "Loss Pred.":
                aggregated_key = ESTIMATOR_CONVERSION_DICT["risk_values"]
            elif method_name == "DUQ":
                aggregated_key = ESTIMATOR_CONVERSION_DICT["duq_values"]
            elif method_name == "Mahalanobis":
                aggregated_key = ESTIMATOR_CONVERSION_DICT["mahalanobis_values"]
            elif method_name in ["GP", "SNGP", "Shallow Ens."]:
                aggregated_key = ESTIMATOR_CONVERSION_DICT["jensen_shannon_divergences"]
            else:
                aggregated_key = ESTIMATOR_CONVERSION_DICT.get(
                    args.distributional_estimator
                )

            if aggregated_key is None:
                operator = max
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
            f"results/gp_eu/{prefix.replace('/', '-')}/aggregated.pdf"
        )
        plot_and_save_aggregated(
            "AUROC",
            aggregated_estimators,
            aggregated_estimators_mins_maxs,
            aggregated_save_path,
            args.label_offsets,
            args.offset_values,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the metric for plotting.")
    parser.add_argument("--distributional-estimator", type=str, default=None)
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

    args = parser.parse_args()
    main(args)
