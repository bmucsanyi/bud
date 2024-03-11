cd ../.. && python train.py --batch-size=128 --crop-pct=1 --data-dir=SLURM_TUE --dataset=torch/cifar10 --dataset-id=soft/cifar10 --decay-milestones="60 120 160" --decay-rate=0.2 --ema-momentum=0.999 --epochs=200 --eval-metric=id_eval_duq_values_auroc_hard_bma_correctness --img-size=32 --is-evaluate-gt --is-evaluate-on-test-sets --lambda-gradient-penalty=0.7491732912421397 --log-wandb --loss=duq --lr=0.09134276827750153 --mean="0.4914 0.4822 0.4465" --method=duq --model=wide_resnet26_10_cifar_preact --momentum=0.9 --no-prefetcher --num-classes=10 --num-hidden-features=-1 --ood-transforms-eval="gaussian_noise shot_noise impulse_noise defocus_blur frosted_glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic pixelate jpeg" --opt=nesterov --padding=2 --rbf-length-scale=0.1 --sched=multistep --seed=0 --smoothing=0 --std="0.2023 0.1994 0.2010" --warmup-epochs=1 --warmup-lr=0.01 --weight-decay=0.0002907591653239365