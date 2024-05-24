import os

ID_TO_METHOD_CIFAR = {
    "wl683ek8": "GP",
    "3vnnnaix": "HET-XL",
    "gypg5gc8": "CE Baseline",
    "9jztoaos": "MC-Dropout",
    "16k5i0w8": "SNGP",
    "03coev3u": "DUQ",
    "6r8nfwqc": "Shallow Ens.",
    "960a6hfa": "Loss Pred.",
    "xsvl0zop": "Corr. Pred.",
    "ymq2jv64": "Deep Ens.",
    "7kksw6rj": "Laplace",
    "swr2k8kf": "Mahalanobis",
    "n85ctsck": "Temperature",
    "f87otin9": "DDU",
    "5bb431gk": "EDL",
    "3lptxghb": "PostNet",
}

ID_TO_METHOD_IMAGENET = {
    "46elax73": "GP",
    "ktze6y0c": "HET-XL",
    "3zt619eq": "CE Baseline",
    "f52l00hb": "MC-Dropout",
    "ew6b0m1x": "SNGP",
    "795iqrk8": "Shallow Ens.",
    "kl7436jj": "Loss Pred.",
    "iskn1vp6": "Corr. Pred.",
    "1nz1l6qj": "Deep Ens.",
    "0qpln50b": "Laplace",
    "mp53zl2m": "Mahalanobis",
    "yxvvtw51": "Temperature",
    "5exmovzc": "DDU",
    "lr19ead6": "EDL",
    "xsd2ro6c": "PostNet",
}

DATASET_CONVERSION_DICT_IMAGENET = {
    "best_id_test": "ImageNet Clean",
    "best_ood_test_soft/imagenetS1": "ImageNet Severity 1",
    "best_ood_test_soft/imagenetS2": "ImageNet Severity 2",
    "best_ood_test_soft/imagenetS3": "ImageNet Severity 3",
    "best_ood_test_soft/imagenetS4": "ImageNet Severity 4",
    "best_ood_test_soft/imagenetS5": "ImageNet Severity 5",
    "best_ood_test_soft/imagenetS1_mixed_soft/imagenet": "ImageNet Clean + Severity 1",
    "best_ood_test_soft/imagenetS2_mixed_soft/imagenet": "ImageNet Clean + Severity 2",
    "best_ood_test_soft/imagenetS3_mixed_soft/imagenet": "ImageNet Clean + Severity 3",
    "best_ood_test_soft/imagenetS4_mixed_soft/imagenet": "ImageNet Clean + Severity 4",
    "best_ood_test_soft/imagenetS5_mixed_soft/imagenet": "ImageNet Clean + Severity 5",
}

DATASET_CONVERSION_DICT_CIFAR = {
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

POSTERIOR_ESTIMATORS = [
    "GP",
    "HET-XL",
    "MC-Dropout",
    "SNGP",
    "Shallow Ens.",
    "Deep Ens.",
    "Laplace",
    "EDL",
    "PostNet",
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
    "gmm_neg_log_densities": r"$u^\text{ddu}$",
    "scaled_inverse_precisions": r"$\text{D-S}$",
    "dempster_shafer_values": r"$\text{D-S}$",
}

GT_LABELS = [
    r"$\text{PU}^\text{b}$",
    r"$\text{B}^\text{b}$",
    r"$\text{AU}^\text{b} + \text{B}^\text{b}$",
    r"$\text{AU}^\text{b}$",
]

ESTIMATORLESS_METRICS = [
    "hard_bma_accuracy",
    "rank_correlation_bma_au_eu",
    "rank_correlation_bregman_au_b_fbar",
    "rank_correlation_bregman_eu_au_hat",
    "rank_correlation_bregman_au_eu",
]


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
