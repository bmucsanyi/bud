#!/usr/bin/env python3
"""ImageNet, CIFAR Training Script

This is intended to be a lean and easily modifiable ImageNet and CIFAR training script
that reproduces ImageNet and CIFAR training results with some of the latest networks and
training techniques. It favours canonical PyTorch and standard Python style over trying
to be able to 'do it all.' That said, it offers quite a few speed and training result
improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
                           and 2024 Bálint Mucsányi (https://github.com/bmucsanyi)
"""

import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import json

from bud import utils
from bud.data import (
    AugMixDataset,
    FastCollateMixup,
    Mixup,
    create_dataset,
    create_loader,
    prepare_n_crop_transform,
    resolve_data_config,
)
from bud.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from bud.losses import (
    BinaryCrossEntropyLoss,
    BMACrossEntropyLoss,
    CorrectnessPredictionLoss,
    DUQLoss,
    EDLLoss,
    UCELoss,
    FBarCrossEntropyLoss,
    JsdCrossEntropyLoss,
    LabelSmoothingCrossEntropyLoss,
    MCInfoNCELoss,
    NonIsotropicVMFLoss,
    LossPredictionLoss,
    SoftTargetCrossEntropyLoss,
)
from bud.models import (
    create_model,
    model_parameters,
    resume_checkpoint,
    safe_model_name,
)
from bud.optimizers import create_optimizer_v2, optimizer_kwargs
from bud.schedulers import create_scheduler_v2, scheduler_kwargs
from bud.utils import ApexScaler, NativeScaler, type_from_string
from bud.wrappers import (
    TemperatureWrapper,
    DUQWrapper,
    DDUWrapper,
    LaplaceWrapper,
    MahalanobisWrapper,
    MCInfoNCEWrapper,
    NonIsotropicvMFWrapper,
    SNGPWrapper,
    PostNetWrapper,
)
from validate import evaluate, evaluate_bulk

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, "compile")


logger = logging.getLogger("train")


def float_list(string):
    return list(map(float, string.split()))


def int_list(string):
    return list(map(int, string.split()))


def string_list(string):
    return string.split()


# The first arg parser parses out only the --config argument, this argument is used to
# Load a yaml file containing key-values that override the defaults for the main parser
# Below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Dataset parameters
group = parser.add_argument_group("Dataset parameters")
group.add_argument("--data-dir", metavar="DIR", help="path to dataset (root dir)")
group.add_argument(
    "--dataset",
    metavar="NAME",
    default="torch/imagenet",
    help='dataset type + name ("<type>/<name>") (default: torch/imagenet)',
)
group.add_argument(
    "--data-dir-id",
    metavar="DIR",
    default=None,
    help="path to id eval + test datasets (root dir) (default: None, uses --data-dir)",
)
group.add_argument(
    "--soft-imagenet-label-dir",
    metavar="DIR",
    default=None,
    help="path to raters.npz and real.json soft ImageNet labels (default: None)",
)
group.add_argument(
    "--dataset-id",
    metavar="NAME",
    default="soft/imagenet",
    help=(
        'id eval + test dataset type + name ("<type>/<name>"), usually the same or a '
        "soft label variant of --dataset (default: soft/imagenet)"
    ),
)
group.add_argument(
    "--data-dir-zero-shot",
    metavar="DIR",
    default=None,
    help=("path to zero-shot datasets (root dir) (default: None, uses --data-dir)"),
)
group.add_argument(
    "--dataset-zero-shot",
    default=[],
    type=string_list,
    help=(
        'list of zero-shot datasets type + name ("<type>/<name>"); '
        "to skip zero-shot evaluation, just provide the same as for --dataset-eval; ",
        "(default: [])",
    ),
)
group.add_argument(
    "--ood-transforms-eval",
    default=[],
    type=string_list,
    help=(
        "list of dataset transforms to be used on the ood eval dataset (default: [])",
    ),
)
group.add_argument(
    "--ood-transforms-test",
    default=[],
    type=string_list,
    help=(
        "list of dataset transforms to be used on the ood test dataset (default: [])",
    ),
)
group.add_argument(
    "--is-evaluate-on-all-splits-id",
    action="store_true",
    default=False,
    help=(
        "whether to evaluate on all splits in-distribution. Only used in soft datasets "
        "to reproduce results of the `pytorch-cifar` repo (default: False)",
    ),
)
group.add_argument(
    "--max-num-id-ood-eval-samples",
    default=100000,
    type=int,
    help=(
        "maximum number of samples in concatenated ID + OOD eval dataset "
        "(default: 100000)"
    ),
)
group.add_argument(
    "--max-num-id-ood-train-samples",
    default=100000,
    type=int,
    help=(
        "maximum number of samples in concatenated ID + OOD train dataset in the "
        "Mahalanobis method (default: 100000)"
    ),
)
group.add_argument(
    "--max-num-id-train-samples",
    default=100000,
    type=int,
    help=(
        "maximum number of samples in (truncated) train dataset for the "
        "DDU method (default: 100000)"
    ),
)
group.add_argument(
    "--max-num-covariance-samples",
    default=100000,
    type=int,
    help=(
        "number of samples to calculate the covariance matrix in the Mahalanobis method "
        "(default: 100000)"
    ),
)
group.add_argument(
    "--is-evaluate-gt",
    action="store_true",
    default=False,
    help="whether to evaluate ground truth uncertainty statistics (default: False)",
)
group.add_argument(
    "--num-repetitions",
    default=20,
    type=int,
    help="number of repetitions when converting soft-label correctness to hard-label "
    "correctness. Example: when the soft-label correctness is 0.8 and "
    "--num-repetitions=10, we get 8 correct and 2 incorrect predictions (default: 20)",
)
group.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (train/validation/test) (default: train)",
)
group.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (train/validation/test) (default: validation)",
)
group.add_argument(
    "--test-split",
    metavar="NAME",
    default="test",
    help="dataset test split ID/OOD (train/validation/test) (default: test)",
)
group.add_argument(
    "--is-evaluate-on-test-sets",
    action="store_true",
    default=False,
    help="evaluate model on the provided test sets (default: False)",
)
group.add_argument(
    "--test-split-zero-shot",
    metavar="NAME",
    default="test",
    help="dataset test split for zero-shot dataset(s). Overrides test-split (default: test)",
)
group.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="allow download of dataset for torch/ and tfds/ datasets that support it",
)
group.add_argument(
    "--class-map",
    default="",
    type=str,
    metavar="FILENAME",
    help='path to class to idx mapping file (default: "")',
)

# Uncertainty method parameters
group = parser.add_argument_group("Method parameters")
group.add_argument(
    "--method",
    default="deterministic",
    type=str,
    metavar="METHOD",
    help='name of uncertainty method (default: "deterministic")',
)
group.add_argument(
    "--num-hidden-features",
    default=256,
    type=int,
    help="number of hidden features in the uncertainty method (default: 256)",
)
group.add_argument(
    "--mlp-depth",
    default=3,
    type=int,
    help=(
        "number of layers in the MLP used by the uncertainty method " "(default: 256)"
    ),
)
group.add_argument(
    "--stopgrad",
    action="store_true",
    default=False,
    help=(
        "whether to stop gradient flow to the model backbone in direct prediction "
        "methods (default: False)"
    ),
)
group.add_argument(
    "--num-hooks",
    default=None,
    type=int,
    help="number of hooks in deep direct prediction methods (default: 5)",
)
group.add_argument(
    "--module-type",
    default=None,
    type=str,
    help=(
        "module type to attach hooks to in direct prediction methods "
        '(default: "torch.nn.ReLU")'
    ),
)
group.add_argument(
    "--module-name-regex",
    default="^(act1|layer[1-4])$",
    type=str,
    help=(
        "module names to attach hooks to in direct prediction methods "
        '(default: "^(act1|layer[1-4])$")'
    ),
)
group.add_argument(
    "--dropout-probability",
    default=0.1,
    type=float,
    help="dropout probability in the dropout method (default: 0.1)",
)
group.add_argument(
    "--is-filterwise-dropout",
    action="store_true",
    default=False,
    help="whether to use filterwise dropout in the dropout method (default: False)",
)
group.add_argument(
    "--num-mc-samples",
    default=10,
    type=int,
    help="number of Monte Carlo samples in the uncertainty method (default: 10)",
)
group.add_argument(
    "--rbf-length-scale",
    default=0.1,
    type=float,
    help="length scale of the RBF kernel in the DUQ method (default: 0.1)",
)
group.add_argument(
    "--ema-momentum",
    default=0.999,
    type=float,
    help=(
        "momentum factor for the exponential moving average in the DUQ method "
        "(default: 0.999)"
    ),
)
group.add_argument(
    "--lambda-gradient-penalty",
    default=0.75,
    type=float,
    help="gradient penalty lambda in the DUQ method (default: 0.75)",
)
group.add_argument(
    "--matrix-rank",
    default=6,
    type=int,
    help="rank of low-rank covariance matrix part in the HET-XL method (default: 6)",
)
group.add_argument(
    "--is-het",
    action="store_true",
    default=False,
    help="whether to use HET instead of HET-XL (default: False)",
)
group.add_argument(
    "--temperature",
    default=1,
    type=float,
    help=("temperature in the HET-XL method (default: 1)"),
)
group.add_argument(
    "--is-last-layer-laplace",
    action="store_true",
    default=False,
    help="whether to use only a last layer Laplace approximation (default: False)",
)
group.add_argument(
    "--pred-type",
    default="glm",
    type=str,
    help='prediction type used by Laplace (default: "glm")',
)
group.add_argument(
    "--prior-optimization-method",
    default="CV",
    type=str,
    help='prior optimization method used by Laplace (default: "CV")',
)
group.add_argument(
    "--hessian-structure",
    default="kron",
    type=str,
    help='Hessian structure method used by Laplace (default: "kron")',
)
group.add_argument(
    "--link-approx",
    default="probit",
    type=str,
    help='link approximation used by Laplace (default: "probit")',
)
group.add_argument(
    "--magnitude",
    default=0.001,
    type=float,
    help="gradient magnitude in the Mahalanobis method (default: 0.001)",
)
group.add_argument(
    "--kappa-pos",
    default=20,
    type=float,
    help="positive kappa value in the MCInfoNCE method (default: 20)",
)
group.add_argument(
    "--initial-average-kappa",
    default=1000,
    type=float,
    help=(
        "initial target average kappa prediction in the MCInfoNCE and nivMF methods "
        "(default: 1000)"
    ),
)
group.add_argument(
    "--num-heads",
    default=10,
    type=int,
    help="number of output heads in the shallow ensemble method (default: 10)",
)
group.add_argument(
    "--is-spectral-normalized",
    action="store_true",
    default=False,
    help="whether to use spectral normalization in the SNGP method (default: False)",
)
group.add_argument(
    "--spectral-normalization-iteration",
    default=1,
    type=int,
    help=(
        "number of iterations in the spectral normalization step of the SNGP method "
        "(default: 1)"
    ),
)
group.add_argument(
    "--spectral-normalization-bound",
    default=6,
    type=float,
    help="bound of the spectral norm in the SNGP method (default: 1)",
)
group.add_argument(
    "--is-batch-norm-spectral-normalized",
    action="store_true",
    default=False,
    help="whether to use spectral normalization in batch norm (default: False)",
)
group.add_argument(
    "--use-tight-norm-for-pointwise-convs",
    action="store_true",
    default=False,
    help="whether to use fully connected spectral normalization for pointwise convs (default: False)",
)
group.add_argument(
    "--num-random-features",
    default=1024,
    type=int,
    help="number of random features in the SNGP method (default: 1024)",
)
group.add_argument(
    "--gp-kernel-scale",
    default=1,
    type=float,
    help="kernel scale in the SNGP method (default: 1)",
)
group.add_argument(
    "--gp-output-bias",
    default=0,
    type=float,
    help="output bias in the SNGP method (default: 0)",
)
group.add_argument(
    "--gp-random-feature-type",
    default="orf",
    type=str,
    help='type of random feature in the SNGP method (default: "orf")',
)
group.add_argument(
    "--is-gp-input-normalized",
    action="store_true",
    default=False,
    help=("whether to normalize the GP's input in the SNGP method (default: False)"),
)
group.add_argument(
    "--gp-cov-momentum",
    default=-1,
    type=float,
    help="momentum term in the SNGP method. If -1, use exact covariance matrix from "
    "last epoch (default: -1)",
)
group.add_argument(
    "--gp-cov-ridge-penalty",
    default=1.0,
    type=float,
    help="ridge penalty for the precision matrix before inverting it (default: 1.0)",
)
group.add_argument(
    "--gp-input-dim",
    default=128,
    type=int,
    help="input dimension to the GP (if > 0, use random projection) (default: 128)",
)
group.add_argument(
    "--postnet-latent-dim",
    default=6,
    type=int,
    help="latent dimensionality in PostNet (default: 6)",
)
group.add_argument(
    "--postnet-num-density-components",
    default=6,
    type=int,
    help="number of density components in PostNet's flow (default: 6)",
)
group.add_argument(
    "--postnet-is-batched",
    action="store_true",
    default=False,
    help=("whether the flow in PostNet is batched (default: False)"),
)
group.add_argument(
    "--is-reset-classifier",
    action="store_true",
    default=False,
    help="whether to reset the classifier layer before training (default: False)",
)
group.add_argument(
    "--is-temperature-scaled",
    action="store_true",
    default=False,
    help=(
        "whether to use temperature scaling for the Temperature and DDU methods "
        "(default: False)"
    ),
)

# Loss parameters
group = parser.add_argument_group("Loss parameters")
group.add_argument(
    "--loss",
    default="cross-entropy",
    type=str,
    help='loss for training (default: "cross-entropy")',
)
group.add_argument(
    "--lambda-uncertainty-loss",
    default=0.01,
    type=float,
    help=(
        "multiplier of the uncertainty loss when calculating the total loss for the "
        "correctness and loss prediction methods (default: 0.01)"
    ),
)
group.add_argument(
    "--is-top5",
    action="store_true",
    default=False,
    help=(
        "whether to use top-5 accuracy to train the correctness prediction method "
        "(default: False)"
    ),
)
group.add_argument(
    "--is-detach",
    action="store_true",
    default=False,
    help=(
        "whether to detach the task loss before calculating the uncertainty loss in the "
        "loss prediction method (default: False)"
    ),
)

# Model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--model",
    default="resnet50",
    type=str,
    metavar="MODEL",
    help='name of model to train (default: "resnet50")',
)
group.add_argument(
    "--weight-paths",
    default=[],
    type=string_list,
    help="list of weight paths for the deep ensemble method (default: [])",
)
group.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="start with pretrained version of specified network (if available)",
)
group.add_argument(
    "--use-pretrained",
    action="store_true",
    default=False,
    help="use pretrained nets **for ensembling** (default: False)",
)
group.add_argument(
    "--freeze-backbone",
    action="store_true",
    default=False,
    help="whether to freeze the model backbone",
)
group.add_argument(
    "--freeze-classifier",
    action="store_true",
    default=False,
    help="whether to freeze the classifier",
)
group.add_argument(
    "--freeze-wrapper",
    action="store_true",
    default=False,
    help="whether to freeze the uncertainty wrapper",
)
group.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="initialize model from this checkpoint (default: none)",
)
group.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="resume full model and optimizer state from checkpoint (default: none)",
)
group.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
group.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (model default if None)",
)
group.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help=(
        "global pool type, one of (fast, avg, max, avgmax, avgmaxc). "
        "(default: None => model default)"
    ),
)
group.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="image size (default: None => model default)",
)
group.add_argument(
    "--in-chans",
    type=int,
    default=None,
    metavar="N",
    help="image input channels (default: None => 3)",
)
group.add_argument(
    "--padding",
    type=int,
    default=2,
    help="padding for the CIFAR training set (default: 2)",
)
group.add_argument(
    "--input-size",
    default=None,
    type=int_list,
    metavar="N N N",
    help=(
        "input all image dimensions (d h w, e.g. --input-size 3 224 224) ",
        "(default: None => model default)",
    ),
)
group.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="input image center crop percent (for validation only)",
)
group.add_argument(
    "--mean",
    type=float_list,
    default=None,
    metavar="MEAN",
    help="override mean pixel value of dataset",
)
group.add_argument(
    "--std",
    type=float_list,
    default=None,
    metavar="STD",
    help="override std of dataset",
)
group.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="image resize interpolation type (overrides model)",
)
group.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size for training (default: 256)",
)
group.add_argument(
    "--accumulation-steps",
    type=int,
    default=1,
    help=(
        "number of batches to accumulate before making an optimizer step "
        "(to simulate a larger batch size)"
    ),
)
group.add_argument(
    "-vb",
    "--validation-batch-size",
    type=int,
    default=None,
    metavar="N",
    help="validation batch size override (default: None)",
)
group.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="use channels_last memory layout",
)
group.add_argument(
    "--fuser",
    default="",
    type=str,
    help="select jit fuser; one of ('', 'te', 'old', 'nvfuser')",
)
group.add_argument(
    "--grad-checkpointing",
    action="store_true",
    default=False,
    help="enable gradient checkpointing through model blocks/stages",
)
group.add_argument(
    "--fast-norm",
    default=False,
    action="store_true",
    help="enable experimental fast-norm",
)
group.add_argument("--model-kwargs", default={}, action=utils.ParseKwargs, type=str)

# Scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="torch.jit.script the full model",
)
scripting_group.add_argument(
    "--torchcompile",
    nargs="?",
    type=str,
    default=None,
    const="inductor",
    help="enable compilation w/ specified backend (default: inductor)",
)

# Optimizer parameters
group = parser.add_argument_group("Optimizer parameters")
group.add_argument(
    "--opt",
    default="lamb",
    type=str,
    metavar="OPTIMIZER",
    help='optimizer (default: "lamb")',
)
group.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="optimizer epsilon (default: None => use opt default)",
)
group.add_argument(
    "--opt-betas",
    default=None,
    type=float_list,
    metavar="BETA",
    help="optimizer betas (default: None => use opt default)",
)
group.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="optimizer momentum (default: 0.9)",
)
group.add_argument(
    "--weight-decay", type=float, default=2e-5, help="weight decay (default: 2e-5)"
)
group.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None => no clipping)",
)
group.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='gradient clipping mode; one of ("norm", "value", "agc") (default: "norm")',
)
group.add_argument(
    "--layer-decay",
    type=float,
    default=None,
    help="layer-wise learning rate decay (default: None)",
)
group.add_argument("--opt-kwargs", nargs="*", default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group("Learning rate schedule parameters")
group.add_argument(
    "--sched",
    type=str,
    default="cosine",
    metavar="SCHEDULER",
    help='LR scheduler (default: "cosine")',
)
group.add_argument(
    "--sched-on-updates",
    action="store_true",
    default=False,
    help="apply LR scheduler step on update instead of epoch end",
)
group.add_argument(
    "--lr",
    type=float,
    default=None,
    metavar="LR",
    help="learning rate, overrides lr-base if set (default: None)",
)
group.add_argument(
    "--lr-base",
    type=float,
    default=0.001,
    metavar="LR",
    help=(
        "base learning rate: lr = lr_base * global_batch_size / base_size "
        "(default: 0.001)"
    ),
)
group.add_argument(
    "--lr-base-size",
    type=int,
    default=256,
    metavar="DIV",
    help="base learning rate batch size (divisor) (default: 256)",
)
group.add_argument(
    "--lr-base-scale",
    type=str,
    default="",
    metavar="SCALE",
    help=(
        'base learning rate vs batch_size scaling ("linear", "sqrt") '
        '(default: "" => based on opt)'
    ),
)
group.add_argument(
    "--lr-noise",
    type=float_list,
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages (default: None)",
)
group.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
group.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std (default: 1.0)",
)
group.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
group.add_argument(
    "--lr-cycle-decay",
    type=float,
    default=0.5,
    metavar="MULT",
    help="amount to decay each learning rate cycle (default: 0.5)",
)
group.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit, cycles enabled if > 1 (default: 1)",
)
group.add_argument(
    "--lr-k-decay",
    type=float,
    default=1.0,
    help="learning rate k-decay for cosine/poly (default: 1.0)",
)
group.add_argument(
    "--warmup-lr",
    type=float,
    default=1e-4,
    metavar="LR",
    help="warmup learning rate (default: 1e-4)",
)
group.add_argument(
    "--min-lr",
    type=float,
    default=0,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (default: 0)",
)
group.add_argument(
    "--epochs",
    type=int,
    default=128,
    metavar="N",
    help="number of epochs to train (default: 128)",
)
group.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help=(
        "epoch repeat multiplier (number of times to repeat dataset epoch per train "
        "epoch) (default: 0.0 => no repeats)"
    ),
)
group.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts) (default: None => start at 1)",
)
group.add_argument(
    "--decay-milestones",
    default=[80, 110],
    type=int_list,
    metavar="MILESTONES",
    help=(
        "list of decay epoch indices for multistep lr; must be increasing "
        "(default: [80, 110])"
    ),
)
group.add_argument(
    "--decay-epochs",
    type=float,
    default=90,
    metavar="N",
    help="epoch interval to decay LR (default: 90)",
)
group.add_argument(
    "--warmup-epochs",
    type=int,
    default=0,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports (default: 0)",
)
group.add_argument(
    "--warmup-prefix",
    action="store_true",
    default=False,
    help="exclude warmup period from decay schedule (default: False)",
),
group.add_argument(
    "--cooldown-epochs",
    type=int,
    default=0,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends (default: 0)",
)
group.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10)",
)
group.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
group = parser.add_argument_group("Augmentation and regularization parameters")
group.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help=(
        "disable all training augmentation, override other train aug args "
        "(default: False)"
    ),
)
group.add_argument(
    "--scale",
    type=float_list,
    default=[0.08, 1.0],
    metavar="PCT",
    help="random resize scale (default: [0.08, 1.0])",
)
group.add_argument(
    "--ratio",
    type=float_list,
    default=[3 / 4, 4 / 3],
    metavar="RATIO",
    help="random resize aspect ratio (default: [0.75, 1.33])",
)
group.add_argument(
    "--hflip",
    type=float,
    default=0.5,
    help="horizontal flip training aug probability (default: 0.5)",
)
group.add_argument(
    "--vflip",
    type=float,
    default=0,
    help="vertical flip training aug probability (default: 0)",
)
group.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="color jitter factor (default: 0.4)",
)
group.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='use AutoAugment policy ("v0", "original") (default: None)',
),
group.add_argument(
    "--aug-repeats",
    type=float,
    default=0,
    help="number of augmentation repetitions (distributed training only) (default: 0)",
)
group.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="number of augmentation splits (valid: 0 or >= 2) (default: 0)",
)
group.add_argument(
    "--jsd-loss",
    action="store_true",
    default=False,
    help=(
        "enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits` "
        "(default: False)"
    ),
)
group.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="enable BCE loss w/ Mixup/CutMix use",
)
group.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="threshold for binarizing softened BCE targets (default: None => disabled)",
)
group.add_argument(
    "--reprob",
    type=float,
    default=0,
    metavar="PCT",
    help="random erase prob (default: 0)",
)
group.add_argument(
    "--remode", type=str, default="pixel", help='random erase mode (default: "pixel")'
)
group.add_argument(
    "--recount", type=int, default=1, help="random erase count (default: 1)"
)
group.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="do not random erase first (clean) augmentation split (default: False)",
)
group.add_argument(
    "--mixup",
    type=float,
    default=0,
    help="mixup alpha, mixup enabled if > 0 (default: 0)",
)
group.add_argument(
    "--cutmix",
    type=float,
    default=0,
    help="cutmix alpha, cutmix enabled if > 0 (default: 0)",
)
group.add_argument(
    "--cutmix-minmax",
    type=float_list,
    default=None,
    help=(
        "cutmix min/max ratio, overrides alpha and enables cutmix if set "
        "(default: None)"
    ),
)
group.add_argument(
    "--mixup-prob",
    type=float,
    default=1,
    help=(
        "probability of performing mixup or cutmix when either/both is enabled "
        "(default: 1)"
    ),
)
group.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help=(
        "probability of switching to cutmix when both mixup and cutmix enabled "
        "(default: 0.5)"
    ),
)
group.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help=(
        'how to apply mixup/cutmix params (per "batch", "pair", or "elem") '
        '(default: "batch")'
    ),
)
group.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="turn off mixup after this epoch, disabled if 0 (default: 0)",
)
group.add_argument(
    "--smoothing",
    type=float,
    default=0.1,
    help="label smoothing (default: 0.1)",
)
group.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='training interpolation ("random", "bilinear", "bicubic") (default: "random")',
)
group.add_argument(
    "--drop", type=float, default=0, metavar="PCT", help="dropout rate (default: 0)"
)
group.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="drop connect rate, DEPRECATED, use drop-path (default: None)",
)
group.add_argument(
    "--drop-path",
    type=float,
    default=None,
    metavar="PCT",
    help="drop path rate (default: None)",
)
group.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="drop block rate (default: None)",
)
group.add_argument(
    "--uce-regularization-factor",
    type=float,
    default=1e-5,
    help="UCE regularization factor for PostNets (default: 1e-5)",
)
group.add_argument(
    "--edl-start-epoch",
    type=int,
    default=1,
    help="start epoch for the EDL flatness regularizer (default: 1)",
)
group.add_argument(
    "--edl-scaler",
    type=float,
    default=1,
    help="scaler for the EDL flatness regularizer (default: 1)",
)
group.add_argument(
    "--edl-activation",
    type=str,
    default="exp",
    help='EDL final activation function (default: "exp")',
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group(
    "Batch norm parameters", "Only works with gen_efficientnet based models currently."
)
group.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None) (default: None)",
)
group.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None) (default: None)",
)
group.add_argument(
    "--sync-bn",
    action="store_true",
    default=False,
    help="enable NVIDIA Apex or Torch synchronized BatchNorm (default: False)",
)
group.add_argument(
    "--dist-bn",
    type=str,
    default="reduce",
    help=(
        "distribute BatchNorm stats between nodes after each epoch "
        '("broadcast", "reduce", "") (default: "reduce")'
    ),
)
group.add_argument(
    "--split-bn",
    action="store_true",
    default=False,
    help="enable separate BN layers per augmentation split (default: False)",
)

# Misc
group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
group.add_argument(
    "--worker-seeding",
    type=str,
    default="all",
    help='worker seed mode (default: "all")',
)
group.add_argument(
    "--log-interval",
    type=int,
    default=50,
    metavar="N",
    help="how many batches to wait before logging training status (default: 50)",
)
group.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint (default: 0)",
)
group.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
group.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
group.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input batches every log interval for debugging (default: False)",
)
group.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help=(
        "use NVIDIA Apex AMP or Native AMP for mixed precision training "
        "(default: True)"
    ),
)
group.add_argument(
    "--amp-dtype",
    default="float16",
    type=str,
    help="lower precision AMP dtype (default: float16)",
)
group.add_argument(
    "--amp-impl",
    default="native",
    type=str,
    help='AMP impl to use, "native" or "apex" (default: "native")',
)
group.add_argument(
    "--no-ddp-bb",
    action="store_true",
    default=False,
    help="force broadcast buffers for native DDP to off (default: False)",
)
group.add_argument(
    "--synchronize-step",
    action="store_true",
    default=False,
    help="whether to call torch.cuda.synchronize() end of each step",
)
group.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help=(
        "pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU "
        "(default: False)"
    ),
)
group.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher (default: False)",
)
group.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help='path to output folder (default: "" => current dir)',
)
group.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help='name of train experiment, name of sub-folder for output (default: "")',
)
group.add_argument(
    "--eval-metric",
    default="id_eval_one_minus_expected_max_probs_auroc_hard_bma_correctness",
    type=str,
    metavar="EVAL_METRIC",
    help=(
        "metric to track for early stopping/checkpoint saving "
        '(default: "id_eval_one_minus_expected_max_probs_auroc_hard_bma_correctness")'
    ),
)
group.add_argument(
    "--decreasing",
    action="store_true",
    default=False,
    help="whether eval-metric is decreasing (default: False)",
)
group.add_argument(
    "--best-save-start-epoch",
    type=int,
    default=0,
    help=(
        "epoch index from which best model according to eval metric is saved "
        "(default: 0)"
    ),
)
group.add_argument("--local-rank", default=0, type=int)
group.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help=(
        "use the multi-epochs-loader to save time at the beginning of every epoch "
        "(default: 0)"
    ),
)
group.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb (default: False)",
)
group.add_argument(
    "--wandb-key",
    type=str,
    default="",
    help="wandb API key",
)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # Defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    if args.data_dir_id is None:
        args.data_dir_id = args.data_dir

    if args.data_dir_zero_shot is None:
        args.data_dir_zero_shot = args.data_dir

    # Detect a special code that tells us to use the local node storage.
    SLURM_TUE_PATH = (
        f"/host/scratch_local/{os.environ.get('SLURM_JOB_USER')}-"
        f"{os.environ.get('SLURM_JOBID')}/datasets"
    )

    if args.data_dir == "SLURM_TUE":
        args.data_dir = SLURM_TUE_PATH

    if args.data_dir_id == "SLURM_TUE":
        args.data_dir_id = SLURM_TUE_PATH

    if args.data_dir_zero_shot == "SLURM_TUE":
        args.data_dir_zero_shot = SLURM_TUE_PATH

    if args.soft_imagenet_label_dir == "SLURM_TUE":
        args.soft_imagenet_label_dir = SLURM_TUE_PATH

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    time_start_setup = datetime.now()
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("CUDA is not available.")

    args.prefetcher = not args.no_prefetcher
    device = utils.init_distributed_device(args)
    if args.distributed:
        logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            if not args.wandb_key:
                with open("wandb_key.json") as f:
                    args.wandb_key = json.load(f)["key"]

            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb.init(project="bias", name=args.experiment, config=args)
        else:
            logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    # Resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == "apex":
            assert has_apex, "AMP impl specified as APEX but APEX is not installed."
            use_amp = "apex"
            assert args.amp_dtype == "float16"
        else:
            assert (
                has_native_amp
            ), "Please update PyTorch to a version with native AMP (or use APEX)."
            use_amp = "native"
            assert args.amp_dtype in ("float16", "bfloat16")
        if args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        model_name=args.model,
        model_wrapper_name=args.method,
        pretrained=args.pretrained,
        scriptable=args.torchscript,
        weight_paths=args.weight_paths,
        num_hidden_features=args.num_hidden_features,
        is_reset_classifier=args.is_reset_classifier,
        mlp_depth=args.mlp_depth,
        stopgrad=args.stopgrad,
        num_hooks=args.num_hooks,
        module_type=type_from_string(args.module_type),
        module_name_regex=args.module_name_regex,
        dropout_probability=args.dropout_probability,
        is_filterwise_dropout=args.is_filterwise_dropout,
        num_mc_samples=args.num_mc_samples,
        rbf_length_scale=args.rbf_length_scale,
        ema_momentum=args.ema_momentum,
        matrix_rank=args.matrix_rank,
        is_het=args.is_het,
        temperature=args.temperature,
        is_last_layer_laplace=args.is_last_layer_laplace,
        pred_type=args.pred_type,
        prior_optimization_method=args.prior_optimization_method,
        hessian_structure=args.hessian_structure,
        link_approx=args.link_approx,
        magnitude=args.magnitude,
        initial_average_kappa=args.initial_average_kappa,
        num_heads=args.num_heads,
        is_spectral_normalized=args.is_spectral_normalized,
        spectral_normalization_iteration=args.spectral_normalization_iteration,
        spectral_normalization_bound=args.spectral_normalization_bound,
        is_batch_norm_spectral_normalized=args.is_batch_norm_spectral_normalized,
        use_tight_norm_for_pointwise_convs=args.use_tight_norm_for_pointwise_convs,
        num_random_features=args.num_random_features,
        gp_kernel_scale=args.gp_kernel_scale,
        gp_output_bias=args.gp_output_bias,
        gp_random_feature_type=args.gp_random_feature_type,
        is_gp_input_normalized=args.is_gp_input_normalized,
        gp_cov_momentum=args.gp_cov_momentum,
        gp_cov_ridge_penalty=args.gp_cov_ridge_penalty,
        gp_input_dim=args.gp_input_dim,
        postnet_latent_dim=args.postnet_latent_dim,
        postnet_num_density_components=args.postnet_num_density_components,
        postnet_is_batched=args.postnet_is_batched,
        edl_activation=args.edl_activation,
        use_pretrained=args.use_pretrained,
        checkpoint_path=args.initial_checkpoint,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        **args.model_kwargs,
    )
    logger.info(str(model))

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        num_params = sum([m.numel() for m in model.parameters()])
        logger.info(
            f"Model {safe_model_name(args.model)} created, param count: {num_params}"
        )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=utils.is_primary(args)
    )

    # Setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # Enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # Freeze model backbone
    if args.freeze_backbone or args.freeze_classifier or args.freeze_wrapper:
        model.set_grads(
            backbone_requires_grad=not args.freeze_backbone,
            classifier_requires_grad=not args.freeze_classifier,
            wrapper_requires_grad=not args.freeze_wrapper,
        )

    # Move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # Setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ""  # Disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == "apex":
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            logger.info(
                "Converted model to use Synchronized BatchNorm. "
                "WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) "
                "while sync-bn enabled."
            )

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    # if not args.lr:
    if args.lr is None:
        global_batch_size = args.batch_size * args.world_size * args.accumulation_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = (
                "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
            )
        if args.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate "
                f"({args.lr_base}) and effective global batch size "
                f"({global_batch_size}) with {args.lr_base_scale} scaling."
            )

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # Setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # Do nothing
    loss_scaler = None
    if use_amp == "apex":
        assert device.type == "cuda"
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        try:
            amp_autocast = partial(
                torch.autocast, device_type=device.type, dtype=amp_dtype
            )
        except (AttributeError, TypeError):
            # Fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == "cuda"
            amp_autocast = torch.cuda.amp.autocast
        if device.type == "cuda" and amp_dtype == torch.float16:
            # Loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if utils.is_primary(args):
            logger.info("AMP not enabled. Training in float32.")

    # Optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # Setup distributed training
    if args.distributed:
        if has_apex and use_amp == "apex":
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb
            )
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # Torch compile should be done after DDP
        assert has_compile, (
            "A version of torch w/ torch.compile() is required for --compile, possibly "
            "a nightly."
        )
        # model = torch.compile(model, backend=args.torchcompile)
        model.model = torch.compile(model.model)
    # Create the train dataset
    dataset_train = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
    )

    # Create the eval datasets
    num_eval_workers = 1

    if args.ood_transforms_test == []:
        args.ood_transforms_test = args.ood_transforms_eval

    dataset_id_eval = create_dataset(
        name=args.dataset_id,
        root=args.data_dir_id,
        label_root=args.soft_imagenet_label_dir,
        is_evaluate_on_all_splits_id=args.is_evaluate_on_all_splits_id,
        split=args.val_split,
        download=args.dataset_download,
        class_map=args.class_map,
        batch_size=args.batch_size,
        is_training=False,
    )

    dataset_id_eval_hard = create_dataset(
        name=args.dataset_id,
        root=args.data_dir_id,
        label_root=args.soft_imagenet_label_dir,
        is_evaluate_on_all_splits_id=args.is_evaluate_on_all_splits_id,
        split=args.val_split,
        download=args.dataset_download,
        class_map=args.class_map,
        batch_size=args.batch_size,
        is_training=False,
    )

    def hard_target_transform(target):
        if isinstance(target, np.ndarray):  # Soft dataset
            return target[-1]  # Last entry contains hard label

    dataset_id_eval_hard.target_transform = hard_target_transform

    if args.ood_transforms_eval:
        dataset_locations_ood_eval = {}
        for severity in range(1, 6):
            dataset_locations_ood_eval[f"{args.dataset_id}S{severity}"] = (
                args.data_dir_id,
                num_eval_workers,
            )

        dataset_locations_ood_test = {}
        for severity in range(1, 6):
            dataset_locations_ood_test[f"{args.dataset_id}S{severity}"] = (
                args.data_dir_id,
                num_eval_workers,
            )

    dataset_locations_zero_shot_test = {}
    for dataset in args.dataset_zero_shot:
        dataset_locations_zero_shot_test[dataset] = (
            args.data_dir_zero_shot,
            num_eval_workers,
        )

    if args.ood_transforms_eval:
        datasets_ood_eval = {}
        for name, (location, num_workers) in dataset_locations_ood_eval.items():
            dataset = create_dataset(
                name=name[:-2],
                root=location,
                label_root=args.soft_imagenet_label_dir,
                split=args.val_split,
                download=args.dataset_download,
                class_map=args.class_map,
                batch_size=args.batch_size,
                is_training=False,
            )
            datasets_ood_eval[name] = (dataset, num_workers)

    dataset_id_test = create_dataset(
        name=args.dataset_id,
        root=args.data_dir_id,
        label_root=args.soft_imagenet_label_dir,
        split=args.test_split,
        download=args.dataset_download,
        class_map=args.class_map,
        batch_size=args.batch_size,
        is_training=False,
    )

    if args.ood_transforms_eval:
        datasets_ood_test = {}
        for name, (location, num_workers) in dataset_locations_ood_test.items():
            dataset = create_dataset(
                name=name[:-2],
                root=location,
                label_root=args.soft_imagenet_label_dir,
                split=args.test_split,
                download=args.dataset_download,
                class_map=args.class_map,
                batch_size=args.batch_size,
                is_training=False,
            )
            datasets_ood_test[name] = (dataset, num_workers)

    datasets_zero_shot_test = {}
    for name, (location, num_workers) in dataset_locations_zero_shot_test.items():
        dataset = create_dataset(
            name=name,
            root=location,
            label_root=args.soft_imagenet_label_dir,
            split=args.test_split_zero_shot,
            download=args.dataset_download,
            class_map=args.class_map,
            batch_size=args.batch_size,
            is_training=False,
        )
        datasets_zero_shot_test[name] = (dataset, num_workers)

    # Setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        if args.prefetcher:
            assert (
                not num_aug_splits
            )  # Collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # Wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # Create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]

    loader_train = create_loader(
        dataset_train,
        dataset_name=args.dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        padding=args.padding,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config["mean"],  # from --mean
        std=data_config["std"],  # from --std
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
        prepare_n_crop_transform=(
            prepare_n_crop_transform if args.method == "mcinfonce" else None
        ),
    )

    if isinstance(model, PostNetWrapper):
        model.calculate_sample_counts(loader_train)

    loader_id_eval = create_loader(
        dataset_id_eval,
        dataset_name=args.dataset_id,
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=num_eval_workers,
        distributed=args.distributed,
        crop_pct=data_config["crop_pct"],
        pin_memory=args.pin_mem,
        device=device,
    )

    loader_id_eval_hard = create_loader(
        dataset_id_eval_hard,
        dataset_name=args.dataset_id,
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=num_eval_workers,
        distributed=args.distributed,
        crop_pct=data_config["crop_pct"],
        pin_memory=args.pin_mem,
        device=device,
    )

    if args.ood_transforms_eval:
        loaders_ood_eval = {}
        for name, (dataset, num_workers) in datasets_ood_eval.items():
            loaders_ood_eval[name] = create_loader(
                dataset,
                dataset_name=name,
                input_size=data_config["input_size"],
                batch_size=args.validation_batch_size or args.batch_size,
                is_training=False,
                use_prefetcher=args.prefetcher,
                interpolation=data_config["interpolation"],
                mean=data_config["mean"],
                std=data_config["std"],
                num_workers=num_eval_workers,
                distributed=args.distributed,
                crop_pct=data_config["crop_pct"],
                pin_memory=args.pin_mem,
                device=device,
                ood_transforms=args.ood_transforms_eval,
                severity=int(name[-1]),
            )

    loader_id_test = create_loader(
        dataset_id_test,
        dataset_name=args.dataset_id,
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=num_eval_workers,
        distributed=args.distributed,
        crop_pct=data_config["crop_pct"],
        pin_memory=args.pin_mem,
        device=device,
    )

    if args.ood_transforms_eval:
        loaders_ood_test = {}
        for name, (dataset, num_workers) in datasets_ood_test.items():
            loaders_ood_test[name] = create_loader(
                dataset,
                dataset_name=name,
                input_size=data_config["input_size"],
                batch_size=args.validation_batch_size or args.batch_size,
                is_training=False,
                use_prefetcher=args.prefetcher,
                interpolation=data_config["interpolation"],
                mean=data_config["mean"],
                std=data_config["std"],
                num_workers=num_eval_workers,
                distributed=args.distributed,
                crop_pct=data_config["crop_pct"],
                pin_memory=args.pin_mem,
                device=device,
                ood_transforms=args.ood_transforms_test,
                severity=int(name[-1]),
            )

    loaders_zero_shot_test = {}
    for name, (dataset, num_workers) in datasets_zero_shot_test.items():
        loaders_zero_shot_test[name] = create_loader(
            dataset,
            dataset_name=name,
            input_size=data_config["input_size"],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=num_eval_workers,
            distributed=args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=args.pin_mem,
            device=device,
        )

    # Initialize uncertainty module
    if (
        isinstance(model, (MCInfoNCEWrapper, NonIsotropicvMFWrapper))
        and args.initial_checkpoint == ""
    ):
        model.initialize_average_kappa(
            loader_train, amp_autocast, device if not args.prefetcher else None
        )

    # Setup loss function
    if args.loss == "cross-entropy":
        if args.jsd_loss:
            assert num_aug_splits > 1, "JSD only valid with aug splits set"
            train_loss_fn = JsdCrossEntropyLoss(
                num_splits=num_aug_splits, smoothing=args.smoothing
            )
        elif mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse,
            # soft targets
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropyLoss(
                    target_threshold=args.bce_target_thresh
                )
            else:
                train_loss_fn = SoftTargetCrossEntropyLoss()
        elif args.smoothing:
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropyLoss(
                    smoothing=args.smoothing, target_threshold=args.bce_target_thresh
                )
            else:
                train_loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "bma-cross-entropy":
        train_loss_fn = BMACrossEntropyLoss()
    elif args.loss == "fbar-cross-entropy":
        train_loss_fn = FBarCrossEntropyLoss()
    elif args.loss == "correctness-prediction":
        train_loss_fn = CorrectnessPredictionLoss(
            args.lambda_uncertainty_loss, args.freeze_backbone, args.is_top5
        )
    elif args.loss == "duq":
        train_loss_fn = DUQLoss()
    elif args.loss == "mcinfonce":
        train_loss_fn = MCInfoNCELoss(args.kappa_pos, args.num_mc_samples)
    elif args.loss == "non-isotropic-vmf":
        train_loss_fn = NonIsotropicVMFLoss()
    elif args.loss == "loss-prediction":
        train_loss_fn = LossPredictionLoss(
            args.lambda_uncertainty_loss, args.is_detach, args.freeze_backbone
        )
    elif args.loss == "edl":
        train_loss_fn = EDLLoss(
            num_batches=len(loader_train),
            num_classes=args.num_classes,
            start_epoch=args.edl_start_epoch,
            scaler=args.edl_scaler,
        )
    elif args.loss == "uce":
        train_loss_fn = UCELoss(regularization_factor=args.uce_regularization_factor)
    else:
        raise NotImplementedError(f"--loss {args.loss} is not implemented.")
    train_loss_fn = train_loss_fn.to(device=device)

    # Setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_eval_metric = float("inf") if args.decreasing else -float("inf")
    best_eval_metrics = None
    best_test_metrics = None
    best_epoch = None
    saver = None
    output_dir = None

    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = utils.get_outdir(
            args.output if args.output else "./output/train", exp_name
        )

        logger.info(f"Output directory is {output_dir}")

        decreasing = args.decreasing
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    # Setup learning rate schedule and starting epoch
    updates_per_epoch = (
        len(loader_train) + args.accumulation_steps - 1
    ) // args.accumulation_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # A specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        logger.info(f"Scheduled epochs: {num_epochs}.")
        logger.info(
            f'LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.'
        )

    time_start_epoch = datetime.now()
    logger.info(
        f"Setup took {(time_start_epoch - time_start_setup).total_seconds()} seconds"
    )

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            if args.lr > 0 and epoch != num_epochs:
                train_metrics = train_one_epoch(
                    epoch,
                    model,
                    loader_train,
                    optimizer,
                    train_loss_fn,
                    args,
                    device=device,
                    lr_scheduler=lr_scheduler,
                    saver=saver,
                    output_dir=output_dir,
                    amp_autocast=amp_autocast,
                    loss_scaler=loss_scaler,
                    mixup_fn=mixup_fn,
                )
            elif args.lr == 0 and epoch == 0:  # Post-hoc method
                logger.info("Learning rate is 0, skipping training epoch.")
                train_metrics = None
            elif args.lr > 0 and epoch == num_epochs and args.is_evaluate_on_test_sets:
                best_save_path = os.path.join(
                    saver.checkpoint_dir, "model_best" + saver.extension
                )
                checkpoint = torch.load(best_save_path, map_location="cpu")
                state_dict = checkpoint["state_dict"]
                model.load_state_dict(state_dict, strict=True)
            else:
                break

            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                if utils.is_primary(args):
                    logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == "reduce")

            if args.ood_transforms_eval:
                update_post_hoc_method(
                    model,
                    loader_train,
                    loader_id_eval_hard,
                    loaders_ood_eval[f"{args.dataset_id}S2"],
                    args,
                )

            eval_metrics = evaluate(
                model=model,
                loader=loader_id_eval,
                device=device,
                amp_autocast=amp_autocast,
                key_prefix="id_eval",
                temp_folder=output_dir,
                is_same_task=True,
                is_upstream=True,
                args=args,
            )

            logger.info(f"{eval_metric}: {eval_metrics[eval_metric]}")

            is_new_best = (args.lr == 0 and epoch == 0) or (
                epoch >= args.best_save_start_epoch
                and (
                    (decreasing and eval_metrics[eval_metric] < best_eval_metric)
                    or (
                        (not decreasing)
                        and eval_metrics[eval_metric] > best_eval_metric
                    )
                )
            )

            if is_new_best:
                best_eval_metric = eval_metrics[eval_metric]
                best_eval_metrics = eval_metrics

            if args.is_evaluate_on_test_sets and (
                (args.lr > 0 and epoch == num_epochs) or (args.lr == 0 and epoch == 0)
            ):
                model.eval()

                if isinstance(model, DDUWrapper):
                    model.fit_gmm(loader_train, args.max_num_id_train_samples)

                if (
                    isinstance(model, (DDUWrapper, TemperatureWrapper))
                    and args.is_temperature_scaled
                ):
                    model.set_temperature_loader(loader_id_eval_hard)

                logger.info(f"Testing best model at epoch {epoch}.")
                # Only for the best model track the test scores
                best_test_metrics = evaluate(
                    model=model,
                    loader=loader_id_test,
                    device=device,
                    amp_autocast=amp_autocast,
                    key_prefix="id_test",
                    temp_folder=output_dir,
                    is_same_task=True,
                    is_upstream=True,
                    args=args,
                )

                if args.ood_transforms_eval:
                    best_test_metrics.update(
                        evaluate_bulk(
                            model=model,
                            loaders=loaders_ood_test,
                            device=device,
                            amp_autocast=amp_autocast,
                            key_prefix="ood_test",
                            temp_folder=output_dir,
                            is_same_task=True,
                            is_upstream=False,
                            args=args,
                        )
                    )

                if len(loaders_zero_shot_test) > 0:
                    best_test_metrics.update(
                        evaluate_bulk(
                            model=model,
                            loaders=loaders_zero_shot_test,
                            device=device,
                            amp_autocast=amp_autocast,
                            key_prefix="zero_shot_test",
                            temp_folder=output_dir,
                            is_same_task=False,
                            is_upstream=False,
                            args=args,
                        )
                    )

            if output_dir is not None:
                lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                utils.update_summary(
                    filename=os.path.join(output_dir, "summary.csv"),
                    epoch=epoch,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                    best_eval_metrics=best_eval_metrics,
                    best_test_metrics=best_test_metrics,
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if (
                saver is not None
                and epoch < num_epochs
                and epoch >= args.best_save_start_epoch
            ):
                # Save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric
                )

            if lr_scheduler is not None and num_epochs > 1:
                # Step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            time_end_epoch = datetime.now()
            logger.info(
                f"Epoch {epoch} took "
                f"{(time_end_epoch - time_start_epoch).total_seconds()} seconds"
            )
            time_start_epoch = time_end_epoch

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        logger.info(f"*** Best metric: {best_metric} (epoch {best_epoch})")


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    device=torch.device("cuda"),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accumulation_steps = args.accumulation_steps
    last_accumulation_steps = len(loader) % accumulation_steps
    updates_per_epoch = (len(loader) + accumulation_steps - 1) // accumulation_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accumulate = len(loader) - last_accumulation_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0

    if isinstance(model, SNGPWrapper) and args.gp_cov_momentum < 0:
        model.classifier[-1].reset_covariance_matrix()

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accumulation_steps == 0
        update_idx = batch_idx // accumulation_steps
        if batch_idx >= last_batch_idx_to_accumulate:
            accumulation_steps = last_accumulation_steps

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # Multiply by accumulation steps to get equivalent to full update
        data_time_m.update(accumulation_steps * (time.time() - data_start_time))

        if isinstance(model, DUQWrapper):
            input.requires_grad_(True)
            target = F.one_hot(target, model.num_classes).float()

        def forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)

                if isinstance(model, DUQWrapper):
                    gradient_penalty = calc_gradient_penalty(input, output)
                    loss += args.lambda_gradient_penalty * gradient_penalty

            if accumulation_steps > 1:
                loss /= accumulation_steps
            return loss

        def backward(loss):
            if loss_scaler is not None:
                loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(
                        model, exclude_head="agc" in args.clip_mode
                    ),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(
                                model, exclude_head="agc" in args.clip_mode
                            ),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

                    if isinstance(model, DUQWrapper):
                        input.requires_grad_(False)

                        with torch.no_grad():
                            model.eval()
                            model.update_centroids(input, target)
                            model.train()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = forward()
                backward(loss)
        else:
            loss = forward()
            backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accumulation_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()

        if args.synchronize_step and device.type == "cuda":
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accumulation_steps, input.size(0))
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                logger.info(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100 * update_idx / (updates_per_epoch - 1):>3.0f}%)]  "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and ((update_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def calc_gradients_input(x, pred):
    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=x,
        grad_outputs=torch.ones_like(pred),
        retain_graph=True,  # Graph still needed for loss backprop
    )[0]

    gradients = gradients.flatten(start_dim=1)

    return gradients


def calc_gradient_penalty(x, pred):
    gradients = calc_gradients_input(x, pred)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two-sided penalty
    gradient_penalty = (grad_norm - 1).square().mean()

    return gradient_penalty


def update_post_hoc_method(
    model, loader_train, loader_id_eval_hard, loader_ood_eval, args
):
    if isinstance(model, LaplaceWrapper):
        assert (
            loader_id_eval_hard is not None
        ), "For Laplace approximation, the ID eval loader has to be specified."
        model.eval()
        model.perform_laplace_approximation(loader_train, loader_id_eval_hard)
    elif isinstance(model, MahalanobisWrapper):
        assert (
            loader_id_eval_hard is not None and loader_ood_eval is not None
        ), "For the Mahalanobis method, the ID and OOD eval loaders have to be specified."
        torch.set_grad_enabled(mode=False)
        model.eval()
        model.train_logistic_regressor(
            loader_train,
            loader_id_eval_hard,
            loader_ood_eval,
            args.max_num_covariance_samples,
            args.max_num_id_ood_train_samples,
        )
        torch.set_grad_enabled(mode=True)


if __name__ == "__main__":
    main()
