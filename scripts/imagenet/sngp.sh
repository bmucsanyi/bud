cd ../.. && python train.py --accumulation-steps=16 --amp --amp-dtype=float16 --batch-size=128 --color-jitter=0 --crop-pct=0.875 --data-dir=/host/scratch_local/datasets/ImageNet2012 --dataset=torch/imagenet --dataset-id=soft/imagenet --epochs=50 --eval-metric=id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness --gp-cov-momentum=-1 --gp-cov-ridge-penalty=1 --gp-input-dim=-1 --gp-kernel-scale=1 --gp-output-bias=0 --gp-random-feature-type=orf --img-size=224 --is-evaluate-gt --is-evaluate-on-test-sets --is-spectral-normalized --log-wandb --loss=cross-entropy --lr-base=0.009635962824120265 --mean="0.485 0.456 0.406" --method=sngp --model=resnet50 --num-classes=1000 --num-mc-samples=1000 --num-random-features=1024 --ood-transforms-eval="gaussian_noise shot_noise impulse_noise defocus_blur frosted_glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic pixelate jpeg" --opt=adamw --pretrained --sched=cosine --seed=0 --smoothing=0 --sngp-version=original --soft-imagenet-label-dir=SLURM_TUE --spectral-normalization-bound=6 --spectral-normalization-iteration=1 --std="0.229 0.224 0.225" --test-split=validation --warmup-epochs=5 --warmup-lr=1e-05 --weight-decay=9.761574307950633e-05