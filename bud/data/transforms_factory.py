""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2019 Ross Wightman
"""
import math

import torch
from torchvision import transforms

from bud.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from bud.data.constants import (
    CIFAR10_DEFAULT_MEAN,
    CIFAR10_DEFAULT_STD,
    DEFAULT_CROP_PCT,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from bud.data.ood_transforms_cifar import OOD_TRANSFORM_DICT_CIFAR
from bud.data.ood_transforms_imagenet import OOD_TRANSFORM_DICT_IMAGENET
from bud.data.random_erasing import RandomErasing
from bud.data.transforms import (
    CenterCropOrPad,
    RandomResizedCropAndInterpolation,
    Resize,
    ResizeKeepRatio,
    ToNumpy,
    str_to_interp_mode,
    str_to_pil_interp,
)


def transforms_noaug_cifar_train(
    img_size=32,
    interpolation="bilinear",
    use_prefetcher=False,
    mean=CIFAR10_DEFAULT_MEAN,
    std=CIFAR10_DEFAULT_STD,
):
    if interpolation == "random":
        # Random interpolation not supported with no-aug
        interpolation = "bilinear"

    tfl = []

    if (isinstance(img_size, tuple) and img_size != (32, 32)) or (
        isinstance(img_size, int) and img_size != 32
    ):
        tfl += [Resize(img_size, interpolation)]

    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return transforms.Compose(tfl)


def transforms_noaug_imagenet_train(
    img_size=224,
    interpolation="bilinear",
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    if interpolation == "random":
        # Random interpolation not supported with no-aug
        interpolation = "bilinear"
    tfl = [
        transforms.Resize(img_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # Prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    return transforms.Compose(tfl)


def transforms_cifar_train(
    img_size=32,
    interpolation="bilinear",
    padding=4,
    hflip=0.5,
    use_prefetcher=False,
    mean=CIFAR10_DEFAULT_MEAN,
    std=CIFAR10_DEFAULT_STD,
):
    tfl = []

    if (isinstance(img_size, tuple) and img_size != (32, 32)) or (
        isinstance(img_size, int) and img_size != 32
    ):
        tfl += [
            Resize(img_size, interpolation),
        ]

    tfl += [
        transforms.RandomCrop(img_size, padding=padding),
    ]

    if hflip > 0:
        tfl += [transforms.RandomHorizontalFlip(p=hflip)]

    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return transforms.Compose(tfl)


def transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
    force_color_jitter=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # Default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # Default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        # Color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not (force_color_jitter or "3a" in auto_augment)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = str_to_pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            aa_params["translate_pct"] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

    if color_jitter is not None and not disable_color_jitter:
        # Color jitter is enabled when not using AA or when forced
        if isinstance(color_jitter, (list, tuple)):
            # Color jitter should be a 3-tuple/list if spec
            # brightness/contrast/saturation or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # If it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # Prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
        if re_prob > 0.0:
            final_tfl.append(
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device="cpu",
                )
            )

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_cifar_eval(
    img_size=32,
    interpolation="bilinear",
    use_prefetcher=False,
    mean=CIFAR10_DEFAULT_MEAN,
    std=CIFAR10_DEFAULT_STD,
    ood_transforms=None,
    severity=0,
):
    ood_transforms = ood_transforms or []

    tfl = []

    if ood_transforms and severity > 0:
        stochastic_ood_transform = StochasticOODTransform(
            ood_transforms, severity, dataset_name="cifar"
        )
        tfl += [stochastic_ood_transform]

    if (isinstance(img_size, tuple) and img_size != (32, 32)) or (
        isinstance(img_size, int) and img_size != 32
    ):
        tfl += [
            Resize(img_size, interpolation),
        ]

    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return CustomCompose(tfl)


def transforms_imagenet_eval(
    img_size=224,
    crop_pct=None,
    crop_mode=None,
    interpolation="bilinear",
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    ood_transforms=None,
    severity=0,
):
    ood_transforms = ood_transforms or []

    tfl = []

    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        scale_size = tuple([math.floor(x / crop_pct) for x in img_size])
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = (scale_size, scale_size)

    if crop_mode == "squash":
        # Squash mode scales each edge to 1/pct of target, then crops
        # Aspect ratio is not preserved, no img lost if crop_pct == 1.0
        tfl += [
            transforms.Resize(
                scale_size, interpolation=str_to_interp_mode(interpolation)
            ),
            transforms.CenterCrop(img_size),
        ]
    elif crop_mode == "border":
        # Scale the longest edge of image to 1/pct of target edge, add borders to pad,
        # then crop, no image lost if crop_pct == 1.0
        fill = [round(255 * v) for v in mean]
        tfl += [
            ResizeKeepRatio(scale_size, interpolation=interpolation, longest=1.0),
            CenterCropOrPad(img_size, fill=fill),
        ]
    else:
        # Default crop model is center
        # Aspect ratio is preserved, crops center within image, no borders are added,
        # image is lost
        if scale_size[0] == scale_size[1]:
            # Simple case, use torchvision built-in Resize w/ shortest edge mode
            # (scalar size arg)
            tfl += [
                transforms.Resize(
                    scale_size[0], interpolation=str_to_interp_mode(interpolation)
                )
            ]
        else:
            # Resize shortest edge to matching target dim for non-square target
            tfl += [ResizeKeepRatio(scale_size)]
        tfl += [transforms.CenterCrop(img_size)]

    # Add stochastic OOD transformations
    if ood_transforms and severity > 0:
        stochastic_ood_transform = StochasticOODTransform(
            ood_transforms, severity, dataset_name="imagenet"
        )
        tfl += [stochastic_ood_transform]

    if use_prefetcher:
        # Prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]

    return CustomCompose(tfl)


class CustomCompose(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, rng=None):
        for t in self.transforms:
            if isinstance(t, StochasticOODTransform):
                img = t(img, rng)
            else:
                img = t(img)

        return img


class StochasticOODTransform:
    def __init__(self, ood_transforms, severity, dataset_name):
        assert any(
            name in dataset_name for name in ["cifar", "imagenet"]
        ), "Corruption transforms only implemented for CIFAR-10(H) and ImageNet"

        self.ood_transforms = ood_transforms
        self.severity = severity

        if "cifar" in dataset_name:
            self.transform_dict = OOD_TRANSFORM_DICT_CIFAR
        elif "imagenet" in dataset_name:
            self.transform_dict = OOD_TRANSFORM_DICT_IMAGENET

    def __call__(self, img, rng):
        idx = rng.integers(low=0, high=len(self.ood_transforms))
        return self.transform_dict[self.ood_transforms[idx]](img, self.severity, rng)


def create_transform(
    input_size,
    dataset_name="imagenet",
    padding=2,
    is_training=False,
    use_prefetcher=False,
    no_aug=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    crop_pct=None,
    crop_mode=None,
    tf_preprocessing=False,
    separate=False,
    ood_transforms=None,
    severity=0,
):
    ood_transforms = ood_transforms or []

    if ood_transforms and severity > 0:
        assert not is_training, "OOD transformations cannot be applied during training"

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from bud.data.tf_preprocessing import TfPreprocessTransform

        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation
        )
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            if "imagenet" in dataset_name:
                transform = transforms_noaug_imagenet_train(
                    img_size,
                    interpolation=interpolation,
                    use_prefetcher=use_prefetcher,
                    mean=mean,
                    std=std,
                )
            elif "cifar" in dataset_name:
                transform = transforms_noaug_cifar_train(
                    img_size,
                    interpolation=interpolation,
                    use_prefetcher=use_prefetcher,
                    mean=mean,
                    std=std,
                )
            else:
                raise ValueError(
                    "Only CIFAR and ImageNet training transforms are supported"
                )
        elif is_training:
            if "imagenet" in dataset_name:
                transform = transforms_imagenet_train(
                    img_size,
                    scale=scale,
                    ratio=ratio,
                    hflip=hflip,
                    vflip=vflip,
                    color_jitter=color_jitter,
                    auto_augment=auto_augment,
                    interpolation=interpolation,
                    use_prefetcher=use_prefetcher,
                    mean=mean,
                    std=std,
                    re_prob=re_prob,
                    re_mode=re_mode,
                    re_count=re_count,
                    re_num_splits=re_num_splits,
                    separate=separate,
                )
            elif "cifar" in dataset_name:
                transform = transforms_cifar_train(
                    img_size,
                    interpolation=interpolation,
                    padding=padding,
                    hflip=hflip,
                    use_prefetcher=use_prefetcher,
                    mean=mean,
                    std=std,
                )
            else:
                raise ValueError(
                    "Only CIFAR and ImageNet training transforms are supported"
                )
        else:
            assert (
                not separate
            ), "Separate transforms not supported for validation preprocessing"
            if "imagenet" in dataset_name:
                transform = transforms_imagenet_eval(
                    img_size,
                    interpolation=interpolation,
                    use_prefetcher=use_prefetcher,
                    mean=mean,
                    std=std,
                    crop_pct=crop_pct,
                    crop_mode=crop_mode,
                    ood_transforms=ood_transforms,
                    severity=severity,
                )
            elif "cifar" in dataset_name:
                transform = transforms_cifar_eval(
                    img_size,
                    interpolation=interpolation,
                    use_prefetcher=use_prefetcher,
                    mean=mean,
                    std=std,
                    ood_transforms=ood_transforms,
                    severity=severity,
                )
            else:
                raise ValueError(
                    "Only CIFAR and ImageNet eval transforms are supported"
                )

    return transform
