import os

ID_TO_METHOD_CIFAR = {
    "82wiia5a": "GP",
    "8pzbl9zq": "HET-XL",
    "mj4gt28n": "CE Baseline",
    "nwmia4kf": "MC-Dropout",
    "b1dd9bjf": "SNGP",
    "y0pqcyo0": "DUQ",
    "k3v4wzua": "Shallow Ens.",
    "t3j6wcsa": "Loss Pred.",
    "ymlbxdms": "Corr. Pred.",
    "yw72v367": "Deep Ens.",
    "7irimi02": "Laplace",
    "aeb5oky6": "Mahalanobis",
    "5j5qcw9l": "Temperature",
    "ipcewyua": "DDU",
    "3l8nkci8": "HET",
    "ihcciqqt": "EDL",
    "c3wpoy10": "PostNet",
}

# ID_TO_METHOD_IMAGENET = {
#     "0zh85pjp": "GP",
#     "n6ocb8vt": "HET-XL",
#     "75316qay": "CE Baseline",
#     "iphs7vdj": "MC-Dropout",
#     "5l11sz1l": "SNGP",
#     "50dvkkny": "Shallow Ens.",
#     "qthh97bn": "Loss Pred.",
#     "7bexzi5z": "Corr. Pred.",
#     "oyn8zlw5": "Deep Ens.",
#     "i170wvxa": "Laplace",
#     "iovcgd69": "Mahalanobis",
#     "9mqh7if3": "Temperature",
#     "n5g7bnct": "DDU",
#     "7yusrr4s": "HET",
#     "52ebshff": "EDL",
#     "g9v0j4p4": "PostNet",
# }

# 100 MIX
# ID_TO_METHOD_IMAGENET = {
#     "0zh85pjp": "GP",
#     "n6ocb8vt": "HET-XL",
#     # "75316qay": "CE Baseline",
#     "iphs7vdj": "MC-Dropout",
#     "5l11sz1l": "SNGP",
#     "50dvkkny": "Shallow Ens.",
#     # "qthh97bn": "Loss Pred.",
#     # "7bexzi5z": "Corr. Pred.",
#     "oyn8zlw5": "Deep Ens.",
#     "i170wvxa": "Laplace",
#     # "iovcgd69": "Mahalanobis",
#     # "9mqh7if3": "Temperature",
#     # "n5g7bnct": "DDU",
#     "7yusrr4s": "HET",
#     "52ebshff": "EDL",
#     "g9v0j4p4": "PostNet",
#     "g4k8mb0g": "HetClassNN",
# }

# 100
ID_TO_METHOD_IMAGENET = {
    "dvdfdfzv": "GP",
    "6kx89627": "HET-XL",
    # "75316qay": "CE Baseline",
    "votzk53f": "MC-Dropout",
    "1csly1f3": "SNGP",
    "kp7tdyy9": "Shallow Ens.",
    # "qthh97bn": "Loss Pred.",
    # "7bexzi5z": "Corr. Pred.",
    "ldjmv256": "Deep Ens.",
    "qthwz783": "Laplace",
    # "iovcgd69": "Mahalanobis",
    # "9mqh7if3": "Temperature",
    # "n5g7bnct": "DDU",
    "nojwarq6": "HET",
    "4f111kzj": "EDL",
    "q5gut0ug": "PostNet",
    "g4k8mb0g": "HetClassNN",
}

# 50
# ID_TO_METHOD_IMAGENET = {
#     "3obkxjlc": "GP",
#     "41uvedx1": "HET-XL",
#     # "75316qay": "CE Baseline",
#     "ux6ynd1j": "MC-Dropout",
#     "32m62dpo": "SNGP",
#     "8hzufqpi": "Shallow Ens.",
#     # "qthh97bn": "Loss Pred.",
#     # "7bexzi5z": "Corr. Pred.",
#     "mxes9emi": "Deep Ens.",
#     "22029uug": "Laplace",
#     # "iovcgd69": "Mahalanobis",
#     # "9mqh7if3": "Temperature",
#     # "n5g7bnct": "DDU",
#     "u9ip4uxj": "HET",
#     "2sysq7hv": "EDL",
#     "y5qumdt6": "PostNet",
#     "almjxh6g": "HetClassNN",
# }

# # 10
# ID_TO_METHOD_IMAGENET = {
#     "lfygbvmj": "GP",
#     "8uw1zzlc": "HET-XL",
#     # "75316qay": "CE Baseline",
#     "px0tqxjb": "MC-Dropout",
#     "8eppzh25": "SNGP",
#     "aj00tthk": "Shallow Ens.",
#     # "qthh97bn": "Loss Pred.",
#     # "7bexzi5z": "Corr. Pred.",
#     "o3vltpi9": "Deep Ens.",
#     "0zohaxr9": "Laplace",
#     # "iovcgd69": "Mahalanobis",
#     # "9mqh7if3": "Temperature",
#     # "n5g7bnct": "DDU",
#     "dfh5oiur": "HET",
#     "zk02olfx": "EDL",
#     "melm6pe1": "PostNet",
#     "7niwvx03": "HetClassNN",
# }

DATASET_CONVERSION_DICT_IMAGENET = {
    "best_id_test": "ImageNet Clean",
    # "best_ood_test_soft/imagenetS1": "ImageNet Severity 1",
    "best_ood_test_soft/imagenetS2": "ImageNet Severity 2",
    # "best_ood_test_soft/imagenetS3": "ImageNet Severity 3",
    # "best_ood_test_soft/imagenetS4": "ImageNet Severity 4",
    # "best_ood_test_soft/imagenetS5": "ImageNet Severity 5",
    # "best_ood_test_soft/imagenetS1_mixed_soft/imagenet": "ImageNet Clean + Severity 1",
    "best_ood_test_soft/imagenetS2_mixed_soft/imagenet": "ImageNet Clean + Severity 2",
    # "best_ood_test_soft/imagenetS3_mixed_soft/imagenet": "ImageNet Clean + Severity 3",
    # "best_ood_test_soft/imagenetS4_mixed_soft/imagenet": "ImageNet Clean + Severity 4",
    # "best_ood_test_soft/imagenetS5_mixed_soft/imagenet": "ImageNet Clean + Severity 5",
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
    "HET",
    "HetClassNN",
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
    "dempster_shafer_values": r"$\text{D-S}$",
    "expected_variances_of_logits": r"$\mathbb{E}\left[\text{var }\log \hat{f}\right]$",
    "expected_variances_of_internal_logits": r"$\mathbb{E}\left[\text{var }\log \hat{f}_\text{int}\right]$",
    "expected_variances_of_probs": r"$\mathbb{E}\left[\text{var }f\right]$",
    "expected_variances_of_internal_probs": r"$\mathbb{E}\left[\text{var }f_\text{int}\right]$",
}

# GT_LABELS = [
#     r"$\text{PU}^\text{b}$",
#     r"$\text{B}^\text{b}$",
#     r"$\text{AU}^\text{b} + \text{B}^\text{b}$",
#     r"$\text{AU}^\text{b}$",
# ]

ESTIMATORLESS_METRICS = [
    "hard_bma_accuracy_original",
    "correlation_bma_au_eu",
    "correlation_bma_eu_pu",
    "correlation_bma_au_pu",
    "correlation_bregman_au_b_fbar",
    "correlation_bregman_eu_au_hat",
    "correlation_bregman_au_eu",
    "correlation_kendall_gal_au_eu_prob",
    "rank_correlation_bma_au_eu",
    "rank_correlation_bma_eu_pu",
    "rank_correlation_bma_au_pu",
    "rank_correlation_bregman_au_b_fbar",
    "rank_correlation_bregman_eu_au_hat",
    "rank_correlation_bregman_au_eu",
    "rank_correlation_kendall_gal_au_eu_prob",
    "log_prob_score_hard_bma_aleatoric_original",
    "brier_score_hard_fbar_aleatoric_original",
]

CONSTRAINED_METRICS = [
    "ece_hard_bma_correctness_original",
    "ece_soft_bma_correctness_original",
    "brier_score_hard_bma_correctness_original",
    "brier_score_soft_bma_correctness_original",
    "log_prob_score_hard_bma_correctness_original",
    "log_prob_score_soft_bma_correctness_original",
]


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
