# Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks

## Introduction

This repository contains code for the arXiv preprint ["Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks"](https://arxiv.org/abs/2402.19460) and also serves as a standalone benchmark suite for future methods.

The `bud` repository extends the [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models/) (`timm`) code base with
- implementations of various uncertainty quantification methods as convenient wrapper classes ... (`bud.wrappers`)
- ... and corresponding loss functions (`bud.losses`)
- an extended training loop that supports these methods out of the box (`train.py`)
- a comprehensive evaluation suite for uncertainty quantification methods (`validate.py`)
- support for CIFAR-10 ResNet variants, including Wide ResNets
- plotting utilities to recreate the plots of the preprint
- scripts to reproduce the results of the preprint

If you found the paper or the code useful in your research, please cite our work as
```
@article{mucsanyi2024benchmarking,
  title={Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks},
  author={Mucs{\'a}nyi, B{\'a}lint and Kirchhof, Michael and Oh, Seong Joon},
  journal={arXiv preprint arXiv:2402.19460},
  year={2024}
}
```

If you use the benchmark, please also cite the datasets it uses.

## Installation

### Packages

Install a [Poetry](https://python-poetry.org/) environment for `bud` by running `poetry install` in the root folder.
Switch to the environment's shell by running `poetry shell`.

OOD perturbations use [Wand](https://docs.wand-py.org/en/0.6.10/index.html), a Python binding of [ImageMagick](https://imagemagick.org/index.php). Follow [these instructions](https://docs.wand-py.org/en/0.6.10/guide/install.html) to install ImageMagick. Wand is installed by the `poetry install` command above.

### Datasets

CIFAR-10 is available in `torchvision.datasets` and is downloaded automatically. A local copy of the ImageNet-1k dataset is needed to run the ImageNet experiments.

The CIFAR-10H test dataset can be downloaded from [this link](https://zenodo.org/records/8115942).

The ImageNet-ReaL labels are available in [this GitHub repository](https://github.com/google-research/reassessed-imagenet). The needed files are `raters.npz` and `real.json`.

## Reproducing Results

We provide scripts that reproduce our results.
These are found in the `scripts` folder for both ImageNet and CIFAR-10 and are named after the respective method.

We also provide access to the exact [Singularity container](https://drive.google.com/file/d/1eYClorSZe3FMNFCZiXGLuSQJizMgPNfq/view?usp=sharing) we used in our experiments.
The `singularity_recipe.rcp` file was used to create this container.

To recreate the plots used in the paper, use `plots/imagenet/create_imagenet_plots.sh` for the main paper's ImageNet results and the individual scripts in the `plots` folder for all other results (incl. appendix figures). To use these utilities, you have to specify your `wandb` API key in `wandb_key.json`.