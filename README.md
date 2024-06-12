# Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks

## Introduction

This repository contains code for the NeurIPS 2024 Datasets and Benchmarks submission "Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks" and also serves as a standalone benchmark suite for future methods.

The `bud` repository extends the [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models/) (`timm`) code base (Apache License) with
- implementations of various uncertainty quantification methods as convenient wrapper classes ... (`bud.wrappers`)
- ... and corresponding loss functions (`bud.losses`)
- an extended training loop that supports these methods out of the box (`train.py`)
- a comprehensive evaluation suite for uncertainty quantification methods (`validate.py`)
- support for CIFAR-10 ResNet variants, including Wide ResNets
- plotting utilities to recreate the plots of the preprint
- scripts to reproduce the results of the preprint

If you use the benchmark, please also cite the datasets it uses.

## Installation

### Packages

Install a [Poetry](https://python-poetry.org/) environment for `bud` by running `poetry install` in the root folder.
Switch to the environment's shell by running `poetry shell`.

OOD perturbations use [Wand](https://docs.wand-py.org/en/0.6.10/index.html), a Python binding of [ImageMagick](https://imagemagick.org/index.php). Follow [these instructions](https://docs.wand-py.org/en/0.6.10/guide/install.html) to install ImageMagick. Wand is installed by the `poetry install` command above.

### Datasets

CIFAR-10 (MIT License) is available in `torchvision.datasets` and is downloaded automatically. A local copy of the ImageNet-1k dataset (Terms of Access for Non-Commercial Use) is needed to run the ImageNet experiments.

The CIFAR-10H test dataset (Creative Commons BY-NC-SA 4.0 License) can be downloaded from [this link](https://zenodo.org/records/8115942).

The ImageNet-ReaL labels (Apache License) are available in [this GitHub repository](https://github.com/google-research/reassessed-imagenet). The needed files are `raters.npz` and `real.json`.

## Reproducing Results

We provide scripts that reproduce our results.
These are found in the `scripts` folder for both ImageNet and CIFAR-10 and are named after the respective method. The dataset paths and the weight paths for post-hoc methods must be modified appropriately.

To recreate the plots used in the paper, we will release all code after the anonymity period.

## Hyperparameters

The hyperparameters used in our experiments are listed in the `configs` folder in Markdown format. These match the ones used in the `scripts` folder but are displayed in a more convenient layout.