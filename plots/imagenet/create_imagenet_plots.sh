python plot_half.py "AUROC Correctness" "auroc_hard_bma_correctness" --y-min=0.5 --y-max=1 --correct-auroc --label-offsets Mahalanobis --offset-values 0.1

python plot_half.py "AUROC Soft Correctness" "auroc_soft_bma_correctness" --y-min=0.5 --y-max=1 --correct-auroc --label-offsets Mahalanobis --offset-values 0.1

python plot_half.py "AUROC OOD-ness" "auroc_oodness" --y-min=0.5 --y-max=1 --correct-auroc

python plot_laplace_eu.py "AUROC OOD-ness" "auroc_oodness" --y-min=0.5 --y-max=1 --correct-auroc

python plot_half.py "Accuracy" "hard_bma_accuracy" --y-min=0.5 --y-max=1

python plot_half.py "Cumulative Abstinence Accuracy AUC" "cumulative_hard_bma_abstinence_auc" --y-min=0.5 --y-max=1

python plot_half.py "Log Prob. Correctness" "log_prob_score_hard_bma_correctness" --y-min=-2 --y-max=0

python plot_half.py "Brier Correctness" "brier_score_hard_bma_correctness" --y-min=-2 --y-max=0

python plot_cross.py

python plot_cross_top5.py

python plot_half.py "Rank Corr. with Bregman Aleatoric" "rank_correlation_bregman_au" --y-min=0 --y-max=1 --correct-abs --label-offsets Mahalanobis --offset-values 0.16

python plot_half.py "Rank Corr. with Bregman Bias" "rank_correlation_bregman_b_fbar" --y-min=0 --y-max=1 --correct-abs --label-offsets Mahalanobis --offset-values 0.16

python plot_laplace_au.py "Rank Corr. with Bregman Aleatoric" "rank_correlation_bregman_au" --y-min=0 --y-max=1 --correct-abs --label-offsets Mahalanobis --offset-values 0.16

python plot_disentanglement.py "Rank Corr. IT Epistemic with Aleatoric" "rank_correlation_bma_au_eu" --y-min=0 --y-max=1 --decreasing --correct-abs

python plot_disentanglement.py "Rank Corr. Bregman Aleatoric with Bias" "rank_correlation_bregman_au_b_fbar" --y-min=0 --y-max=1 --decreasing --correct-abs

python plot_correlation_matrix_flatten.py

python plot_cross_ece.py

python plot_half.py "ECE Correctness" "ece_hard_bma_correctness" --y-min=0 --y-max=0.25 --decreasing --label-offsets SNGP GP Dropout "Shallow Ens." "Deep Ens." "Baseline" "HET-XL" Laplace "Corr. Pred." --offset-values 0.02 0.02 0.04 0.06 0.003 0.003 0.003 0.003 0.003

