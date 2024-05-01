import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr
import sys
import json

sys.path.insert(0, "..")

from utils import (
    DATASET_CONVERSION_DICT_CIFAR,
    ESTIMATOR_CONVERSION_DICT,
    create_directory,
)

from tueplots import bundles
from tueplots.constants.color import rgb

plt.rcParams.update(
    bundles.icml2022(family="serif", usetex=True, nrows=1, column="half")
)


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
    "CE Baseline",
    "MC-Dropout",
    "SNGP",
    "DUQ",
    "Shallow Ensemble",
    "Correctness Prediction",
    "Deep Ensemble",
    "Laplace",
]


def plot_and_save(label_x, label_y, metric_x, metric_y, key_x, key_y, save_path):
    """Plots and saves the metric's bar chart with min-max error bars as a PDF."""
    fig, ax = plt.subplots()
    ax.grid(zorder=1)
    # fig.set_figheight(1.8)
    ax.scatter(
        metric_x[key_x],
        metric_y[key_y],
        zorder=2,
        color=rgb.tue_green,
        label="Optimize x-axis metric",
    )
    ax.scatter(
        metric_x[key_x],
        metric_y[key_y],
        zorder=2,
        color=rgb.tue_red,
        label="Optimize y-axis metric",
    )
    ax.spines[["right", "top"]].set_visible(False)

    ax.set_xlabel(f"{label_x}")
    ax.set_ylabel(f"{label_y}")

    # fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_and_save_aggregated(
    label_x,
    label_y,
    data_xs_opt_x,
    data_xs_opt_y,
    data_ys_opt_x,
    data_ys_opt_y,
    save_path,
):
    """Plots and saves the aggregated best values with min-max error bars as a PDF."""
    full_xs_opt_x = []
    full_ys_opt_x = []
    full_xs_opt_y = []
    full_ys_opt_y = []

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.grid(zorder=1)
    ax2.grid(zorder=1)
    # fig.set_figheight(1.8)
    for key in data_xs_opt_x:
        values_x = data_xs_opt_x[key]
        values_y = data_ys_opt_x[key]
        ax1.scatter(values_x, values_y, zorder=2, label=key, s=10)

        full_xs_opt_x.extend(values_x)
        full_ys_opt_x.extend(values_y)

    for key in data_xs_opt_y:
        values_x = data_xs_opt_y[key]
        values_y = data_ys_opt_y[key]
        ax2.scatter(values_x, values_y, zorder=2, label=key, s=10)

        full_xs_opt_y.extend(values_x)
        full_ys_opt_y.extend(values_y)

    handles, labels = [], []
    for ax in (ax1, ax2):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    ax1.spines[["right", "top"]].set_visible(False)
    ax2.spines[["right", "top"]].set_visible(False)
    ax1.set_xlabel(f"{label_x}")
    ax1.set_ylabel(
        f"{label_y} (rcorr: {spearmanr(full_xs_opt_x, full_ys_opt_x)[0]:.4f})"
    )
    ax2.set_xlabel(f"{label_x}")
    ax2.set_ylabel(
        f"{label_y} (rcorr: {spearmanr(full_xs_opt_y, full_ys_opt_y)[0]:.4f})"
    )
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    plt.savefig(save_path)
    plt.close()


def plot_and_save_aggregated_shared_axes(
    label_x,
    label_y,
    data_xs_opt_x,
    data_xs_opt_y,
    data_ys_opt_x,
    data_ys_opt_y,
    save_path,
):
    """Plots and saves the aggregated best values with shared axes and rcorr on the plot."""
    full_xs_opt_x = []
    full_ys_opt_x = []
    full_xs_opt_y = []
    full_ys_opt_y = []

    fig, ax = plt.subplots()
    ax.grid(zorder=1)

    # Get the default color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_iterator = iter(color_cycle)

    legend_handles = {}

    for key in data_xs_opt_x:
        try:
            color = next(color_iterator)  # Get next color in the cycle
        except StopIteration:
            color_iterator = iter(
                [rgb.tue_mauve, rgb.tue_gold, rgb.tue_lightgreen, rgb.tue_gray]
            )

        values_x_opt_x = data_xs_opt_x[key]
        values_y_opt_x = data_ys_opt_x[key]
        values_x_opt_y = data_xs_opt_y[key]
        values_y_opt_y = data_ys_opt_y[key]

        scatter_x = ax.scatter(
            np.mean(values_x_opt_x),
            np.mean(values_y_opt_x),
            color=color,
            zorder=3,
            label=key,
            s=10,
        )
        ax.scatter(
            np.mean(values_x_opt_y),
            np.mean(values_y_opt_y),
            color=color,
            zorder=3,
            s=10,
        )

        legend_handles[key] = scatter_x

        # Draw lines between corresponding points with the same color
        ax.plot(
            [np.mean(values_x_opt_x), np.mean(values_x_opt_y)],
            [np.mean(values_y_opt_x), np.mean(values_y_opt_y)],
            color=color,
            zorder=2,
        )

        full_xs_opt_x.extend(values_x_opt_x)
        full_ys_opt_x.extend(values_y_opt_x)
        full_xs_opt_y.extend(values_x_opt_y)
        full_ys_opt_y.extend(values_y_opt_y)

    rcorr_x = spearmanr(full_xs_opt_x, full_ys_opt_x)[0]
    rcorr_y = spearmanr(full_xs_opt_y, full_ys_opt_y)[0]

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    print(f"{label_x} (rcorr = {round(rcorr_x, 3)})")
    print(f"{label_y} (rcorr = {round(rcorr_y, 3)})")

    fig.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc="center left",
        bbox_to_anchor=(0.6333, 0.479),
        borderpad=0.1,
        # frameon=False,
    )
    plt.savefig(save_path)
    plt.close()


def main(args):
    with open("../../wandb_key.json") as f:
        wandb_key = json.load(f)["key"]

    wandb.login(key=wandb_key)
    api = wandb.Api()

    id_to_method = {
        "2vkuhe38": "GP",
        "3vnnnaix": "HET-XL",
        "gypg5gc8": "CE Baseline",
        "9jztoaos": "MC-Dropout",
        "f32n7c05": "SNGP",
        "03coev3u": "DUQ",
        "6r8nfwqc": "Shallow Ensemble",
        "960a6hfa": "Loss Prediction",
        "xsvl0zop": "Correctness Prediction",
        "ymq2jv64": "Deep Ensemble",
        "gkvfnbup": "Laplace",
        "swr2k8kf": "Mahalanobis",
    }

    suffix_x = args.metric_x
    suffix_y = args.metric_y

    for prefix in DATASET_CONVERSION_DICT_CIFAR:
        create_directory("results")
        create_directory(f"results/{args.metric_x}_{args.metric_y}")
        create_directory(
            f"results/{args.metric_x}_{args.metric_y}/{prefix.replace('/', '-')}"
        )
        data_xs_opt_x = {}
        data_xs_opt_y = {}
        data_ys_opt_x = {}
        data_ys_opt_y = {}

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

            save_path = f"results/{args.metric_x}_{args.metric_y}/{prefix.replace('/', '-')}/{method_name}.pdf"
            metric_x = {
                key: value for key, value in metric_x.items() if "NaN" not in value
            }
            metric_y = {
                key: value for key, value in metric_y.items() if "NaN" not in value
            }

            if not metric_x or not metric_y:
                continue

            if method_name == "Correctness Prediction":
                key_x = "error_probabilities"
            elif method_name == "Loss Prediction":
                key_x = "risk_values"
            elif method_name == "DUQ":
                key_x = "duq_values"
            elif method_name == "Mahalanobis":
                key_x = "mahalanobis_values"
            else:
                key_x = None

            if key_x is None:
                operator = min if args.decreasing_x else max
                means_x = {
                    key: np.mean(metric_x[key])
                    for key in metric_x
                    if "gt" not in key and key in metric_y
                }
                key_x = operator(means_x.items(), key=lambda x: x[1])[0]

            if method_name == "Correctness Prediction":
                key_y = "error_probabilities"
            elif method_name == "Loss Prediction":
                key_y = "risk_values"
            elif method_name == "DUQ":
                key_y = "duq_values"
            elif method_name == "Mahalanobis":
                key_y = "mahalanobis_values"
            else:
                key_y = None

            if key_y is None:
                operator = min if args.decreasing_y else max
                means_y = {
                    key: np.mean(metric_y[key])
                    for key in metric_y
                    if "gt" not in key and key in metric_x
                }
                key_y = operator(means_y.items(), key=lambda x: x[1])[0]

            if key_x not in metric_x or key_y not in metric_y:
                continue

            data_xs_opt_x[method_name] = metric_x[key_x]
            data_xs_opt_y[method_name] = metric_x[key_y]

            data_ys_opt_x[method_name] = metric_y[key_x]
            data_ys_opt_y[method_name] = metric_y[key_y]

            plot_and_save(
                args.label_x,
                args.label_y,
                metric_x,
                metric_y,
                key_x,
                key_y,
                save_path,
            )

        if (
            not data_xs_opt_x
            or not data_xs_opt_y
            or not data_ys_opt_x
            or not data_ys_opt_y
        ):
            continue

        # Save the aggregated plot with min-max error bars
        aggregated_save_path = f"results/{args.metric_x}_{args.metric_y}/{prefix.replace('/', '-')}/aggregated.pdf"
        plot_and_save_aggregated_shared_axes(
            args.label_x,
            args.label_y,
            data_xs_opt_x,
            data_xs_opt_y,
            data_ys_opt_x,
            data_ys_opt_y,
            aggregated_save_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the metric for plotting.")
    parser.add_argument("label_x", type=str, help="Label of x axis.")
    parser.add_argument("label_y", type=str, help="Label of y axis.")
    parser.add_argument("metric_x", type=str, help="Name of the x-axis metric.")
    parser.add_argument("metric_y", type=str, help="Name of the y-axis metric.")
    parser.add_argument(
        "--decreasing-x",
        action="store_true",
        default=False,
        help="Whether the metric is increasing or decreasing.",
    )
    parser.add_argument(
        "--decreasing-y",
        action="store_true",
        default=False,
        help="Whether the metric is increasing or decreasing.",
    )

    args = parser.parse_args()
    main(args)
