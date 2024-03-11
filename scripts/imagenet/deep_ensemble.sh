cd ../.. && python train.py --accumulation-steps=16 --amp --amp-dtype=float16 --batch-size=128 --color-jitter=0 --crop-pct=0.875 --data-dir=/host/scratch_local/datasets/ImageNet2012 --dataset=torch/imagenet --dataset-id=soft/imagenet --epochs=1 --eval-metric=id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness --img-size=224 --is-evaluate-gt --is-evaluate-on-test-sets --log-wandb --loss=cross-entropy --lr-base=0 --mean="0.485 0.456 0.406" --method=deep-ensemble --model=resnet50 --momentum=0.9 --num-classes=1000 --ood-transforms-eval="gaussian_noise shot_noise impulse_noise defocus_blur frosted_glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic pixelate jpeg" --opt=adamw --pretrained --sched=cosine --smoothing=0 --soft-imagenet-label-dir=SLURM_TUE --std="0.229 0.224 0.225" --test-split=validation --warmup-epochs=0 --warmup-lr=0 --weight-decay=0.001 --weight-paths="/path/to/network1 /path/to/network2 /path/to/network3 /path/to/network4 /path/to/network5"