import os

ID_TO_METHOD_CIFAR = {
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

ID_TO_METHOD_IMAGENET = {
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
]


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
