cd ../.. && python train.py --batch-size=128 --best-save-start-epoch=0 --crop-pct=1 --data-dir=SLURM_TUE --dataset=torch/cifar10 --dataset-id=soft/cifar10 --decay-milestones="75 150 200" --decay-rate=0.2 --epochs=250 --eval-metric=id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness --gp-cov-momentum=-1 --gp-cov-ridge-penalty=1 --gp-input-dim=-1 --gp-kernel-scale=1 --gp-output-bias=0 --gp-random-feature-type=orf --img-size=32 --is-evaluate-gt --is-evaluate-on-test-sets --log-wandb --loss=cross-entropy --lr=0.07732882071439598 --mean="0.4914 0.4822 0.4465" --method=sngp --model=wide_resnet26_10_cifar_preact --momentum=0.9 --no-prefetcher --num-classes=10 --num-mc-samples=1000 --num-random-features=1024 --ood-transforms-eval="gaussian_noise shot_noise impulse_noise defocus_blur frosted_glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic pixelate jpeg" --opt=nesterov --padding=2 --sched=multistep --seed=0 --smoothing=0 --std="0.2023 0.1994 0.2010" --warmup-epochs=1 --warmup-lr=0.01 --weight-decay=0.00016445092988226514