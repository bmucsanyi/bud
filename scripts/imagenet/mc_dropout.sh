cd ../.. && python train.py --accumulation-steps=16 --amp --amp-dtype=float16 --batch-size=128 --color-jitter=0 --crop-pct=0.875 --data-dir=<DATA_DIR> --dataset=torch/imagenet --dataset-id=soft/imagenet --dropout-probability=0.05 --epochs=50 --eval-metric=id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness --img-size=224 --is-evaluate-gt --is-evaluate-on-test-sets --is-filterwise-dropout --log-wandb --loss=cross-entropy --lr-base=0.0004292477116017232 --mean="0.485 0.456 0.406" --method=dropout --model=resnet50 --momentum=0.9 --num-classes=1000 --num-mc-samples=10 --ood-transforms-eval="gaussian_noise shot_noise impulse_noise defocus_blur frosted_glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic pixelate jpeg" --opt=adamw --pretrained --sched=cosine --seed=0 --smoothing=0 --soft-imagenet-label-dir=<SOFT_IMAGENET_LABEL_DIR> --std="0.229 0.224 0.225" --test-split=validation --warmup-epochs=5 --warmup-lr=1e-05 --weight-decay=4.0775116368648755e-06