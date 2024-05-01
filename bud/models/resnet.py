"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

CIFAR ResNet variants implemented based on https://github.com/google/uncertainty-baselines

Copyright 2019 Ross Wightman
      and 2024 Bálint Mucsányi
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from bud.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from bud.layers import (
    AvgPool2dSame,
    BlurPool2d,
    DropBlock2d,
    DropPath,
    GroupNorm,
    create_attn,
    create_classifier,
    get_act_layer,
    get_attn,
    get_norm_layer,
)

from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import (
    generate_default_cfgs,
    register_model,
    register_model_deprecations,
)

__all__ = [
    "ResNet",
    "BasicBlock",
    "Bottleneck",
]  # model_registry will add each entrypoint fn to this


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
    ):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock does not support changing base width"
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes,
            first_planes,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            bias=False,
        )
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(
            aa_layer, channels=first_planes, stride=stride, enable=use_aa
        )

        self.conv2 = nn.Conv2d(
            first_planes,
            outplanes,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, "weight", None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
    ):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes,
            width,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=p,
                dilation=first_dilation,
                bias=False,
            ),
            norm_layer(out_channels),
        ]
    )


def downsample_avg(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = (
            AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        )
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(
        *[
            pool,
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            norm_layer(out_channels),
        ]
    )


class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(
            x.shape[0],
            self.out_c - self.in_c,
            x.shape[2],
            x.shape[3],
            device=x.device,
        )
        x = torch.cat((x, pad), dim=1)
        return x


def downsample_ddu(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer=None,
):
    return nn.Sequential(
        AvgPoolShortCut(stride=stride, out_c=out_channels, in_c=in_channels)
    )


def drop_blocks(drop_prob=0.0):
    return [
        None,
        None,
        (
            partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25)
            if drop_prob
            else None
        ),
        (
            partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00)
            if drop_prob
            else None
        ),
    ]


def make_blocks(
    block_fn,
    channels,
    block_repeats,
    inplanes,
    reduce_first=1,
    output_stride=32,
    down_kernel_size=1,
    down_type="conv",
    drop_block_rate=0.0,
    drop_path_rate=0.0,
    **kwargs,
):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(
        zip(channels, block_repeats, drop_blocks(drop_block_rate))
    ):
        stage_name = f"layer{stage_idx + 1}"  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get("norm_layer"),
            )
            if down_type == "conv":
                downsample = downsample_conv(**down_kwargs)
            elif down_type == "avg":
                downsample = downsample_avg(**down_kwargs)
            else:
                downsample = downsample_ddu(**down_kwargs)

        block_kwargs = dict(
            reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs
        )
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = (
                drop_path_rate * net_block_idx / (net_num_blocks - 1)
            )  # stochastic depth linear decay rule
            blocks.append(
                block_fn(
                    inplanes,
                    planes,
                    stride,
                    downsample,
                    first_dilation=prev_dilation,
                    drop_path=DropPath(block_dpr) if block_dpr > 0.0 else None,
                    **block_kwargs,
                )
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(
            dict(num_chs=inplanes, reduction=net_stride, module=stage_name)
        )

    return stages, feature_info


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        in_chans=3,
        output_stride=32,
        global_pool="avg",
        cardinality=1,
        base_width=64,
        stem_width=64,
        stem_type="",
        replace_stem_pool=False,
        block_reduce_first=1,
        down_kernel_size=1,
        down_type="conv",
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        aa_layer=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=0.0,
        zero_init_last=True,
        block_args=None,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            down_type (str): type of projection skip connection between stages/downsample (default conv)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = "deep" in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if "tiered" in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv2d(
                        in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False
                    ),
                    norm_layer(stem_chs[0]),
                    act_layer(inplace=True),
                    nn.Conv2d(
                        stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False
                    ),
                    norm_layer(stem_chs[1]),
                    act_layer(inplace=True),
                    nn.Conv2d(
                        stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False
                    ),
                ]
            )
        else:
            self.conv1 = nn.Conv2d(
                in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module="act1")]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(
                *filter(
                    None,
                    [
                        nn.Conv2d(
                            inplanes,
                            inplanes,
                            3,
                            stride=1 if aa_layer else 2,
                            padding=1,
                            bias=False,
                        ),
                        (
                            create_aa(aa_layer, channels=inplanes, stride=2)
                            if aa_layer is not None
                            else None
                        ),
                        norm_layer(inplanes),
                        act_layer(inplace=True),
                    ],
                )
            )
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(
                        *[
                            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                            aa_layer(channels=inplanes, stride=2),
                        ]
                    )
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            down_type=down_type,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, "zero_init_last"):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r"^conv1|bn1|maxpool",
            blocks=r"^layer(\d+)" if coarse else r"^layer(\d+)\.(\d+)",
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(
                [self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True
            )
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class BasicBlockCIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_type="conv"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if down_type == "conv":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            elif down_type == "ddu":
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(
                        stride=stride, out_c=self.expansion * planes, in_c=in_planes
                    )
                )
            else:
                raise ValueError("Invalid downsample type provided.")

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class BottleneckCIFAR(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_type="conv"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.act3 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if down_type == "conv":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            elif down_type == "ddu":
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(
                        stride=stride, out_c=self.expansion * planes, in_c=in_planes
                    )
                )
            else:
                raise ValueError("Invalid downsample type provided.")

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        block,
        depth,
        width_multiplier=1,
        num_classes=10,
        down_type="conv",
        act="relu",
    ):
        super().__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.num_features = 64 * block.expansion * width_multiplier
        self.down_type = down_type

        assert (
            depth - 2
        ) % 6 == 0, "depth should be 6n+2 (e.g., 20, 32, 44, 56, 110, 1202)"
        n = (depth - 2) // 6

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU() if act == "relu" else nn.LeakyReLU()
        self.layer1 = self.make_layer(block, 16 * width_multiplier, n, stride=1)
        self.layer2 = self.make_layer(block, 32 * width_multiplier, n, stride=2)
        self.layer3 = self.make_layer(block, 64 * width_multiplier, n, stride=2)
        self.global_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        blocks = [
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                down_type=self.down_type,
            )
        ]
        self.in_planes = planes * block.expansion

        for _ in range(num_blocks - 1):
            blocks.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=1,
                    down_type=self.down_type,
                )
            )

        return nn.Sequential(*blocks)

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward_features(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out

    def forward_head(self, x, pre_logits: bool = False):
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)

        return out if pre_logits else self.fc(out)

    def forward(self, x):
        out = self.forward_features(x)
        out = self.forward_head(out)

        return out


class BasicBlockCIFARPreAct(nn.Module):
    """Pre-activation version of the BasicBlock for CIFAR ResNets."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_type="conv"):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            if down_type == "conv":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                )
            elif down_type == "ddu":
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(
                        stride=stride, out_c=self.expansion * planes, in_c=in_planes
                    )
                )
            else:
                raise ValueError("Invalid downsample type provided.")

    def forward(self, x):
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        out += shortcut
        return out


class BottleneckCIFARPreAct(nn.Module):
    """Pre-activation version of the original Bottleneck module for CIFAR ResNets."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_type="conv"):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn3 = nn.BatchNorm2d(planes)
        self.act3 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            if down_type == "conv":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                )
            elif down_type == "ddu":
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(
                        stride=stride, out_c=self.expansion * planes, in_c=in_planes
                    )
                )
            else:
                raise ValueError("Invalid downsample type provided.")

    def forward(self, x):
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        out = self.conv3(self.act3(self.bn3(out)))
        out += shortcut
        return out


class ResNetCIFARPreAct(nn.Module):
    def __init__(
        self,
        block,
        depth,
        width_multiplier=1,
        num_classes=10,
        down_type="conv",
        act="relu",
    ):
        super().__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.num_features = 64 * block.expansion * width_multiplier
        self.down_type = down_type

        assert (
            depth - 2
        ) % 6 == 0, "depth should be 6n+2 (e.g., 20, 32, 44, 56, 110, 1202)"
        n = (depth - 2) // 6

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.make_layer(block, 16 * width_multiplier, n, stride=1)
        self.layer2 = self.make_layer(block, 32 * width_multiplier, n, stride=2)
        self.layer3 = self.make_layer(block, 64 * width_multiplier, n, stride=2)
        self.bn = nn.BatchNorm2d(self.num_features)
        self.act = nn.ReLU() if act == "relu" else nn.LeakyReLU()
        self.global_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        blocks = [
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                down_type=self.down_type,
            )
        ]
        self.in_planes = planes * block.expansion

        for _ in range(num_blocks - 1):
            blocks.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=1,
                    down_type=self.down_type,
                )
            )

        return nn.Sequential(*blocks)

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward_features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.act(out)

        return out

    def forward_head(self, x, pre_logits: bool = False):
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)

        return out if pre_logits else self.fc(out)

    def forward(self, x):
        out = self.forward_features(x)
        out = self.forward_head(out)

        return out


#    DDDD   U   U  QQQQ
#    D   D  U   U Q   Q
#    D   D  U   U Q   Q
#    D   D  U   U Q  QQ
#    D   D  U   U Q   Q
#    D   D  U   U Q   Q
#    DDDD    UUU   QQQQ Q

"""
From: https://github.com/jhjacobsen/invertible-resnet
Which is based on: https://arxiv.org/abs/1811.00995

Soft Spectral Normalization (not enforced, only <= coeff) for Conv2D layers
Based on: Regularisation of Neural Networks by Enforcing Lipschitz Continuity
    (Gouk et al. 2018)
    https://arxiv.org/abs/1804.04368
"""
import torch
from torch.nn.functional import normalize, conv_transpose2d, conv2d


class SpectralNormConv(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(
        self, coeff, input_dim, name="weight", n_power_iterations=1, eps=1e-12
    ):
        self.coeff = coeff
        self.input_dim = input_dim
        self.name = name
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                "got n_power_iterations={}".format(n_power_iterations)
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module, do_power_iteration):
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important bahaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is alreay on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        sigma_log = getattr(module, self.name + "_sigma")  # for logging

        # get settings from conv-module (for transposed convolution)
        stride = module.stride
        padding = module.padding

        if do_power_iteration:
            with torch.no_grad():
                output_padding = 0
                if stride[0] > 1:
                    # Note: the below does not generalize to stride > 2
                    output_padding = 1 - self.input_dim[-1] % 2
                for _ in range(self.n_power_iterations):
                    v_s = conv_transpose2d(
                        u.view(self.out_shape),
                        weight,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    )
                    # Note: out flag for in-place changes
                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)

                    u_s = conv2d(
                        v.view(self.input_dim),
                        weight,
                        stride=stride,
                        padding=padding,
                        bias=None,
                    )
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()
        weight_v = conv2d(
            v.view(self.input_dim), weight, stride=stride, padding=padding, bias=None
        )
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        # enforce spectral norm only as constraint
        factorReverse = torch.max(
            torch.ones(1, device=weight.device), sigma / self.coeff
        )

        # rescaling
        weight = weight / (factorReverse + 1e-5)  # for stability

        # for logging
        sigma_log.copy_(sigma.detach())

        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        assert (
            inputs[0].shape[1:] == self.input_dim[1:]
        ), "Input dims don't match actual input"
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    @staticmethod
    def apply(module, coeff, input_dim, name, n_power_iterations, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormConv(coeff, input_dim, name, n_power_iterations, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            num_input_dim = input_dim[0] * input_dim[1] * input_dim[2] * input_dim[3]
            v = normalize(torch.randn(num_input_dim), dim=0, eps=fn.eps)

            # get settings from conv-module (for transposed convolution)
            stride = module.stride
            padding = module.padding
            # forward call to infer the shape
            u = conv2d(
                v.view(input_dim), weight, stride=stride, padding=padding, bias=None
            )
            fn.out_shape = u.shape
            num_output_dim = (
                fn.out_shape[0] * fn.out_shape[1] * fn.out_shape[2] * fn.out_shape[3]
            )
            # overwrite u with random init
            u = normalize(torch.randn(num_output_dim), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormConvStateDictHook(fn))
        module._register_load_state_dict_pre_hook(
            SpectralNormConvLoadStateDictPreHook(fn)
        )
        return fn


class SpectralNormConvLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        fn = self.fn
        version = local_metadata.get("spectral_norm_conv", {}).get(
            fn.name + ".version", None
        )
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + "_orig"]
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + "_u"]


class SpectralNormConvStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if "spectral_norm_conv" not in local_metadata:
            local_metadata["spectral_norm_conv"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm_conv"]:
            raise RuntimeError(
                "Unexpected key in metadata['spectral_norm_conv']: {}".format(key)
            )
        local_metadata["spectral_norm_conv"][key] = self.fn._version


def spectral_norm_conv(
    module, coeff, input_dim, n_power_iterations, name="weight", eps=1e-12
):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """
    input_dim_4d = torch.Size([1, input_dim[0], input_dim[1], input_dim[2]])
    SpectralNormConv.apply(module, coeff, input_dim_4d, name, n_power_iterations, eps)
    return module


def remove_spectral_norm_conv(module, name="weight"):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNormConv) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))


"""
Obtained from torch.nn.utils.spectral_norm at tag 1.6

Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from torch.nn.modules import Module


class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    # name: str
    # dim: int
    # n_power_iterations: int
    # eps: float

    def __init__(
        self,
        coeff: float = 1.0,
        name: str = "weight",
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        self.coeff = coeff
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                "got n_power_iterations={}".format(n_power_iterations)
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim]
            )
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        sigma_log = getattr(module, self.name + "_sigma")  # for logging
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log.copy_(sigma.detach())

        return weight

    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(
            weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)
        ).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(
        module: Module,
        name: str,
        coeff: float,
        n_power_iterations: int,
        dim: int,
        eps: float,
    ) -> "SpectralNorm":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNorm(coeff, name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormLoadStateDictPreHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        fn = self.fn
        version = local_metadata.get("spectral_norm", {}).get(
            fn.name + ".version", None
        )
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if (
                version is None
                and all(weight_key + s in state_dict for s in ("_orig", "_u", "_v"))
                and weight_key not in state_dict
            ):
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ("_orig", "", "_u"):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + "_orig"]
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + "_u"]
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + "_v"] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if "spectral_norm" not in local_metadata:
            local_metadata["spectral_norm"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm"]:
            raise RuntimeError(
                "Unexpected key in metadata['spectral_norm']: {}".format(key)
            )
        local_metadata["spectral_norm"][key] = self.fn._version


T_module = TypeVar("T_module", bound=Module)


def spectral_norm_fc(
    module: T_module,
    coeff: float = 1.0,
    n_power_iterations: int = 1,
    name: str = "weight",
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> T_module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, coeff, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module: T_module, name: str = "weight") -> T_module:
    r"""Removes the spectral normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
            break

    for k, hook in module._load_state_dict_pre_hooks.items():
        if isinstance(hook, SpectralNormLoadStateDictPreHook) and hook.fn.name == name:
            del module._load_state_dict_pre_hooks[k]
            break

    return module


# Obtained from: https://github.com/meliketoy/wide-resnet.pytorch
# Adapted to match:
# https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class WideBasic(nn.Module):
    def __init__(
        self,
        wrapped_conv,
        input_size,
        in_c,
        out_c,
        stride,
        dropout_rate,
        mod=True,
        batchnorm_momentum=0.01,
    ):
        super().__init__()

        self.mod = mod
        self.bn1 = nn.BatchNorm2d(in_c, momentum=batchnorm_momentum)
        self.conv1 = wrapped_conv(input_size, in_c, out_c, 3, stride)

        self.bn2 = nn.BatchNorm2d(out_c, momentum=batchnorm_momentum)
        self.conv2 = wrapped_conv(math.ceil(input_size / stride), out_c, out_c, 3, 1)
        self.activation = F.leaky_relu if self.mod else F.relu

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        if stride != 1 or in_c != out_c:
            if mod:

                def shortcut(x):
                    x = F.avg_pool2d(x, stride, stride)
                    pad = torch.zeros(
                        x.shape[0],
                        out_c - in_c,
                        x.shape[2],
                        x.shape[3],
                        device=x.device,
                    )
                    x = torch.cat((x, pad), dim=1)
                    return x

                self.shortcut = shortcut
            else:
                # Just use a strided conv
                self.shortcut = wrapped_conv(input_size, in_c, out_c, 1, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.activation(self.bn1(x))
        out = self.conv1(out)
        out = self.activation(self.bn2(out))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        spectral_normalization=True,
        mod=True,
        depth=28,
        widen_factor=10,
        num_classes=None,
        dropout_rate=0.3,
        coeff=3,
        n_power_iterations=1,
        batchnorm_momentum=0.01,
        temp=1.0,
        **kwargs,
    ):
        """
        If the "mod" parameter is set to True, the architecture uses 2 modifications:
        1. LeakyReLU instead of normal ReLU
        2. Average Pooling on the residual connections.
        """
        super().__init__()

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"

        self.dropout_rate = dropout_rate
        self.mod = mod
        self.num_features = 64 * widen_factor

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, shapes, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        strides = [1, 1, 2, 2]
        input_sizes = 32 // np.cumprod(strides)

        self.conv1 = wrapped_conv(input_sizes[0], 3, nStages[0], 3, strides[0])
        self.layer1 = self._wide_layer(nStages[0:2], n, strides[1], input_sizes[0])
        self.layer2 = self._wide_layer(nStages[1:3], n, strides[2], input_sizes[1])
        self.layer3 = self._wide_layer(nStages[2:4], n, strides[3], input_sizes[2])

        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=batchnorm_momentum)
        self.activation = F.leaky_relu if self.mod else F.relu

        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(nStages[3], num_classes)

        nonlinearity = "leaky_relu" if self.mod else "relu"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L17
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=nonlinearity
                )
            elif isinstance(m, nn.Linear):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L21
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=nonlinearity
                )
                nn.init.constant_(m.bias, 0)
        self.feature = None
        self.temp = temp

    def _wide_layer(self, channels, num_blocks, stride, input_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        in_c, out_c = channels

        for stride in strides:
            layers.append(
                WideBasic(
                    self.wrapped_conv,
                    input_size,
                    in_c,
                    out_c,
                    stride,
                    self.dropout_rate,
                    self.mod,
                )
            )
            in_c = out_c
            input_size = math.ceil(input_size / stride)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.flatten(1)
        self.feature = out.clone().detach()

        if self.num_classes is not None:
            out = self.linear(out) / self.temp
        return out


#    DDDD   U   U  QQQQ
#    D   D  U   U Q   Q
#    D   D  U   U Q   Q
#    D   D  U   U Q  QQ
#    D   D  U   U Q   Q
#    D   D  U   U Q   Q
#    DDDD    UUU   QQQQ Q


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (7, 7),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "conv1",
        "classifier": "fc",
        **kwargs,
    }


def _tcfg(url="", **kwargs):
    return _cfg(url=url, **dict({"interpolation": "bicubic"}, **kwargs))


def _ttcfg(url="", **kwargs):
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
                "test_input_size": (3, 288, 288),
                "test_crop_pct": 0.95,
                "origin_url": "https://github.com/huggingface/pytorch-image-models",
            },
            **kwargs,
        ),
    )


def _rcfg(url="", **kwargs):
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
                "crop_pct": 0.95,
                "test_input_size": (3, 288, 288),
                "test_crop_pct": 1.0,
                "origin_url": "https://github.com/huggingface/pytorch-image-models",
                "paper_ids": "arXiv:2110.00476",
            },
            **kwargs,
        ),
    )


def _r3cfg(url="", **kwargs):
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
                "input_size": (3, 160, 160),
                "pool_size": (5, 5),
                "crop_pct": 0.95,
                "test_input_size": (3, 224, 224),
                "test_crop_pct": 0.95,
                "origin_url": "https://github.com/huggingface/pytorch-image-models",
                "paper_ids": "arXiv:2110.00476",
            },
            **kwargs,
        ),
    )


def _gcfg(url="", **kwargs):
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
                "origin_url": "https://cv.gluon.ai/model_zoo/classification.html",
            },
            **kwargs,
        ),
    )


default_cfgs = generate_default_cfgs(
    {
        # ResNet and Wide ResNet trained w/ timm (RSB paper and others)
        "resnet10t.c3_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet10t_176_c3-f3215ab1.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_crop_pct=0.95,
            test_input_size=(3, 224, 224),
            first_conv="conv1.0",
        ),
        "resnet14t.c3_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet14t_176_c3-c4ed2c37.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_crop_pct=0.95,
            test_input_size=(3, 224, 224),
            first_conv="conv1.0",
        ),
        "resnet18.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth",
        ),
        "resnet18.a2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a2_0-b61bd467.pth",
        ),
        "resnet18.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a3_0-40c531c8.pth",
        ),
        "resnet18d.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth",
            first_conv="conv1.0",
        ),
        "resnet34.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a1_0-46f8f793.pth",
        ),
        "resnet34.a2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a2_0-82d47d71.pth",
        ),
        "resnet34.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a3_0-a20cabb6.pth",
            crop_pct=0.95,
        ),
        "resnet34.bt_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth",
        ),
        "resnet34d.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth",
            first_conv="conv1.0",
        ),
        "resnet26.bt_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth",
        ),
        "resnet26d.bt_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth",
            first_conv="conv1.0",
        ),
        "resnet26t.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet26t_256_ra2-6f6fa748.pth",
            first_conv="conv1.0",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            crop_pct=0.94,
            test_input_size=(3, 320, 320),
            test_crop_pct=1.0,
        ),
        "resnet50.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
        ),
        "resnet50.a1h_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            crop_pct=0.9,
            test_input_size=(3, 224, 224),
            test_crop_pct=1.0,
        ),
        "resnet50.a2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a2_0-a2746f79.pth",
        ),
        "resnet50.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a3_0-59cae1ef.pth",
        ),
        "resnet50.b1k_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_b1k-532a802a.pth",
        ),
        "resnet50.b2k_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_b2k-1ba180c1.pth",
        ),
        "resnet50.c1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_c1-5ba5e060.pth",
        ),
        "resnet50.c2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_c2-d01e05b2.pth",
        ),
        "resnet50.d_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_d-f39db8af.pth",
        ),
        "resnet50.ram_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
        ),
        "resnet50.am_in1k": _tcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_am-6c502b37.pth",
        ),
        "resnet50.ra_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ra-85ebb6e5.pth",
        ),
        "resnet50.bt_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/rw_resnet50-86acaeed.pth",
        ),
        "resnet50d.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth",
            first_conv="conv1.0",
        ),
        "resnet50d.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth",
            first_conv="conv1.0",
        ),
        "resnet50d.a2_in1k": _rcfg(
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a2_0-a3adc64d.pth",
            first_conv="conv1.0",
        ),
        "resnet50d.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a3_0-403fdfad.pth",
            first_conv="conv1.0",
        ),
        "resnet50t.untrained": _ttcfg(first_conv="conv1.0"),
        "resnet101.a1h_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth",
        ),
        "resnet101.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1_0-cdcb52a9.pth",
        ),
        "resnet101.a2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a2_0-6edb36c7.pth",
        ),
        "resnet101.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a3_0-1db14157.pth",
        ),
        "resnet101d.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth",
            first_conv="conv1.0",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            crop_pct=0.95,
            test_crop_pct=1.0,
            test_input_size=(3, 320, 320),
        ),
        "resnet152.a1h_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth",
        ),
        "resnet152.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1_0-2eee8a7a.pth",
        ),
        "resnet152.a2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a2_0-b4c6978f.pth",
        ),
        "resnet152.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a3_0-134d4688.pth",
        ),
        "resnet152d.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth",
            first_conv="conv1.0",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            crop_pct=0.95,
            test_crop_pct=1.0,
            test_input_size=(3, 320, 320),
        ),
        "resnet200.untrained": _ttcfg(),
        "resnet200d.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth",
            first_conv="conv1.0",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            crop_pct=0.95,
            test_crop_pct=1.0,
            test_input_size=(3, 320, 320),
        ),
        "wide_resnet50_2.racm_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth",
        ),
        # torchvision resnet weights
        "resnet18.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnet18-5c106cde.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet34.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnet34-333f7ec4.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet50.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnet50-19c8e357.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet50.tv2_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_input_size=(3, 224, 224),
            test_crop_pct=0.965,
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet101.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet101.tv2_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_input_size=(3, 224, 224),
            test_crop_pct=0.965,
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet152.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnet152-b121ed2d.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet152.tv2_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_input_size=(3, 224, 224),
            test_crop_pct=0.965,
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "wide_resnet50_2.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "wide_resnet50_2.tv2_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_input_size=(3, 224, 224),
            test_crop_pct=0.965,
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "wide_resnet101_2.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "wide_resnet101_2.tv2_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_input_size=(3, 224, 224),
            test_crop_pct=0.965,
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        # ResNets w/ alternative norm layers
        "resnet50_gn.a1h_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth",
            crop_pct=0.94,
        ),
        # ResNeXt trained in timm (RSB paper and others)
        "resnext50_32x4d.a1h_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth",
        ),
        "resnext50_32x4d.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1_0-b5a91a1d.pth",
        ),
        "resnext50_32x4d.a2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a2_0-efc76add.pth",
        ),
        "resnext50_32x4d.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a3_0-3e450271.pth",
        ),
        "resnext50_32x4d.ra_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d_ra-d733960d.pth",
        ),
        "resnext50d_32x4d.bt_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth",
            first_conv="conv1.0",
        ),
        "resnext101_32x4d.untrained": _ttcfg(),
        "resnext101_64x4d.c1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnext101_64x4d_c-0d0e0cc0.pth",
        ),
        # torchvision ResNeXt weights
        "resnext50_32x4d.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnext101_32x8d.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnext101_64x4d.tv_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnext50_32x4d.tv2_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_input_size=(3, 224, 224),
            test_crop_pct=0.965,
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnext101_32x8d.tv2_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
            input_size=(3, 176, 176),
            pool_size=(6, 6),
            test_input_size=(3, 224, 224),
            test_crop_pct=0.965,
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        #  ResNeXt models - Weakly Supervised Pretraining on Instagram Hashtags
        #  from https://github.com/facebookresearch/WSL-Images
        #  Please note the CC-BY-NC 4.0 license on these weights, non-commercial use only.
        "resnext101_32x8d.fb_wsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/WSL-Images",
        ),
        "resnext101_32x16d.fb_wsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/WSL-Images",
        ),
        "resnext101_32x32d.fb_wsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/WSL-Images",
        ),
        "resnext101_32x48d.fb_wsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/WSL-Images",
        ),
        #  Semi-Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
        #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
        "resnet18.fb_ssl_yfcc100m_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnet50.fb_ssl_yfcc100m_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        #  Semi-Weakly Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
        #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
        "resnet18.fb_swsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnet50.fb_swsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnext50_32x4d.fb_swsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnext101_32x4d.fb_swsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnext101_32x8d.fb_swsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        "resnext101_32x16d.fb_swsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth",
            license="cc-by-nc-4.0",
            origin_url="https://github.com/facebookresearch/semi-supervised-ImageNet1K-models",
        ),
        #  Efficient Channel Attention ResNets
        "ecaresnet26t.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet26t_ra2-46609757.pth",
            first_conv="conv1.0",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            test_crop_pct=0.95,
            test_input_size=(3, 320, 320),
        ),
        "ecaresnetlight.miil_in1k": _tcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnetlight-75a9c627.pth",
            test_crop_pct=0.95,
            test_input_size=(3, 288, 288),
        ),
        "ecaresnet50d.miil_in1k": _tcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet50d-93c81e3b.pth",
            first_conv="conv1.0",
            test_crop_pct=0.95,
            test_input_size=(3, 288, 288),
        ),
        "ecaresnet50d_pruned.miil_in1k": _tcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet50d_p-e4fa23c2.pth",
            first_conv="conv1.0",
            test_crop_pct=0.95,
            test_input_size=(3, 288, 288),
        ),
        "ecaresnet50t.ra2_in1k": _tcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet50t_ra2-f7ac63c4.pth",
            first_conv="conv1.0",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            test_crop_pct=0.95,
            test_input_size=(3, 320, 320),
        ),
        "ecaresnet50t.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a1_0-99bd76a8.pth",
            first_conv="conv1.0",
        ),
        "ecaresnet50t.a2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a2_0-b1c7b745.pth",
            first_conv="conv1.0",
        ),
        "ecaresnet50t.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a3_0-8cc311f1.pth",
            first_conv="conv1.0",
        ),
        "ecaresnet101d.miil_in1k": _tcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet101d-153dad65.pth",
            first_conv="conv1.0",
            test_crop_pct=0.95,
            test_input_size=(3, 288, 288),
        ),
        "ecaresnet101d_pruned.miil_in1k": _tcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet101d_p-9e74cb91.pth",
            first_conv="conv1.0",
            test_crop_pct=0.95,
            test_input_size=(3, 288, 288),
        ),
        "ecaresnet200d.untrained": _ttcfg(
            first_conv="conv1.0",
            input_size=(3, 256, 256),
            crop_pct=0.95,
            pool_size=(8, 8),
        ),
        "ecaresnet269d.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet269d_320_ra2-7baa55cb.pth",
            first_conv="conv1.0",
            input_size=(3, 320, 320),
            pool_size=(10, 10),
            crop_pct=0.95,
            test_crop_pct=1.0,
            test_input_size=(3, 352, 352),
        ),
        #  Efficient Channel Attention ResNeXts
        "ecaresnext26t_32x4d.untrained": _tcfg(first_conv="conv1.0"),
        "ecaresnext50t_32x4d.untrained": _tcfg(first_conv="conv1.0"),
        #  Squeeze-Excitation ResNets, to eventually replace the models in senet.py
        "seresnet18.untrained": _ttcfg(),
        "seresnet34.untrained": _ttcfg(),
        "seresnet50.a1_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a1_0-ffa00869.pth",
            crop_pct=0.95,
        ),
        "seresnet50.a2_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a2_0-850de0d9.pth",
            crop_pct=0.95,
        ),
        "seresnet50.a3_in1k": _r3cfg(
            hf_hub_id="timm/",
            url="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a3_0-317ecd56.pth",
            crop_pct=0.95,
        ),
        "seresnet50.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth",
        ),
        "seresnet50t.untrained": _ttcfg(first_conv="conv1.0"),
        "seresnet101.untrained": _ttcfg(),
        "seresnet152.untrained": _ttcfg(),
        "seresnet152d.ra2_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet152d_ra2-04464dd2.pth",
            first_conv="conv1.0",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            crop_pct=0.95,
            test_crop_pct=1.0,
            test_input_size=(3, 320, 320),
        ),
        "seresnet200d.untrained": _ttcfg(
            first_conv="conv1.0", input_size=(3, 256, 256), pool_size=(8, 8)
        ),
        "seresnet269d.untrained": _ttcfg(
            first_conv="conv1.0", input_size=(3, 256, 256), pool_size=(8, 8)
        ),
        #  Squeeze-Excitation ResNeXts, to eventually replace the models in senet.py
        "seresnext26d_32x4d.bt_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26d_32x4d-80fa48a3.pth",
            first_conv="conv1.0",
        ),
        "seresnext26t_32x4d.bt_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26tn_32x4d-569cb627.pth",
            first_conv="conv1.0",
        ),
        "seresnext50_32x4d.racm_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext50_32x4d_racm-a304a460.pth",
        ),
        "seresnext101_32x4d.untrained": _ttcfg(),
        "seresnext101_32x8d.ah_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101_32x8d_ah-e6bc4c0a.pth",
        ),
        "seresnext101d_32x8d.ah_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101d_32x8d_ah-191d7b94.pth",
            first_conv="conv1.0",
        ),
        # ResNets with anti-aliasing / blur pool
        "resnetaa50d.sw_in12k_ft_in1k": _ttcfg(
            hf_hub_id="timm/", first_conv="conv1.0", crop_pct=0.95, test_crop_pct=1.0
        ),
        "resnetaa101d.sw_in12k_ft_in1k": _ttcfg(
            hf_hub_id="timm/", first_conv="conv1.0", crop_pct=0.95, test_crop_pct=1.0
        ),
        "seresnextaa101d_32x8d.sw_in12k_ft_in1k_288": _ttcfg(
            hf_hub_id="timm/",
            crop_pct=0.95,
            input_size=(3, 288, 288),
            pool_size=(9, 9),
            test_input_size=(3, 320, 320),
            test_crop_pct=1.0,
            first_conv="conv1.0",
        ),
        "seresnextaa101d_32x8d.sw_in12k_ft_in1k": _ttcfg(
            hf_hub_id="timm/", first_conv="conv1.0", test_crop_pct=1.0
        ),
        "seresnextaa201d_32x8d.sw_in12k_ft_in1k_384": _cfg(
            hf_hub_id="timm/",
            interpolation="bicubic",
            first_conv="conv1.0",
            pool_size=(12, 12),
            input_size=(3, 384, 384),
            crop_pct=1.0,
        ),
        "seresnextaa201d_32x8d.sw_in12k": _cfg(
            hf_hub_id="timm/",
            num_classes=11821,
            interpolation="bicubic",
            first_conv="conv1.0",
            crop_pct=0.95,
            input_size=(3, 320, 320),
            pool_size=(10, 10),
            test_input_size=(3, 384, 384),
            test_crop_pct=1.0,
        ),
        "resnetaa50d.sw_in12k": _ttcfg(
            hf_hub_id="timm/",
            num_classes=11821,
            first_conv="conv1.0",
            crop_pct=0.95,
            test_crop_pct=1.0,
        ),
        "resnetaa50d.d_in12k": _ttcfg(
            hf_hub_id="timm/",
            num_classes=11821,
            first_conv="conv1.0",
            crop_pct=0.95,
            test_crop_pct=1.0,
        ),
        "resnetaa101d.sw_in12k": _ttcfg(
            hf_hub_id="timm/",
            num_classes=11821,
            first_conv="conv1.0",
            crop_pct=0.95,
            test_crop_pct=1.0,
        ),
        "seresnextaa101d_32x8d.sw_in12k": _ttcfg(
            hf_hub_id="timm/",
            num_classes=11821,
            first_conv="conv1.0",
            crop_pct=0.95,
            test_crop_pct=1.0,
        ),
        "resnetblur18.untrained": _ttcfg(),
        "resnetblur50.bt_in1k": _ttcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnetblur50-84f4748f.pth",
        ),
        "resnetblur50d.untrained": _ttcfg(first_conv="conv1.0"),
        "resnetblur101d.untrained": _ttcfg(first_conv="conv1.0"),
        "resnetaa50.a1h_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetaa50_a1h-4cf422b3.pth",
        ),
        "seresnetaa50d.untrained": _ttcfg(first_conv="conv1.0"),
        "seresnextaa101d_32x8d.ah_in1k": _rcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnextaa101d_32x8d_ah-83c8ae12.pth",
            first_conv="conv1.0",
        ),
        # ResNet-RS models
        "resnetrs50.tf_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs50_ema-6b53758b.pth",
            input_size=(3, 160, 160),
            pool_size=(5, 5),
            crop_pct=0.91,
            test_input_size=(3, 224, 224),
            interpolation="bicubic",
            first_conv="conv1.0",
        ),
        "resnetrs101.tf_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs101_i192_ema-1509bbf6.pth",
            input_size=(3, 192, 192),
            pool_size=(6, 6),
            crop_pct=0.94,
            test_input_size=(3, 288, 288),
            interpolation="bicubic",
            first_conv="conv1.0",
        ),
        "resnetrs152.tf_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs152_i256_ema-a9aff7f9.pth",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            crop_pct=1.0,
            test_input_size=(3, 320, 320),
            interpolation="bicubic",
            first_conv="conv1.0",
        ),
        "resnetrs200.tf_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnetrs200_c-6b698b88.pth",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            crop_pct=1.0,
            test_input_size=(3, 320, 320),
            interpolation="bicubic",
            first_conv="conv1.0",
        ),
        "resnetrs270.tf_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs270_ema-b40e674c.pth",
            input_size=(3, 256, 256),
            pool_size=(8, 8),
            crop_pct=1.0,
            test_input_size=(3, 352, 352),
            interpolation="bicubic",
            first_conv="conv1.0",
        ),
        "resnetrs350.tf_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs350_i256_ema-5a1aa8f1.pth",
            input_size=(3, 288, 288),
            pool_size=(9, 9),
            crop_pct=1.0,
            test_input_size=(3, 384, 384),
            interpolation="bicubic",
            first_conv="conv1.0",
        ),
        "resnetrs420.tf_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs420_ema-972dee69.pth",
            input_size=(3, 320, 320),
            pool_size=(10, 10),
            crop_pct=1.0,
            test_input_size=(3, 416, 416),
            interpolation="bicubic",
            first_conv="conv1.0",
        ),
        # gluon resnet weights
        "resnet18.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet18_v1b-0757602b.pth",
        ),
        "resnet34.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet34_v1b-c6d82d59.pth",
        ),
        "resnet50.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1b-0ebe02e2.pth",
        ),
        "resnet101.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1b-3b017079.pth",
        ),
        "resnet152.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1b-c1edb0dd.pth",
        ),
        "resnet50c.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1c-48092f55.pth",
            first_conv="conv1.0",
        ),
        "resnet101c.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1c-1f26822a.pth",
            first_conv="conv1.0",
        ),
        "resnet152c.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1c-a3bb0b98.pth",
            first_conv="conv1.0",
        ),
        "resnet50d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1d-818a1b1b.pth",
            first_conv="conv1.0",
        ),
        "resnet101d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1d-0f9c8644.pth",
            first_conv="conv1.0",
        ),
        "resnet152d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1d-bd354e12.pth",
            first_conv="conv1.0",
        ),
        "resnet50s.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1s-1762acc0.pth",
            first_conv="conv1.0",
        ),
        "resnet101s.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1s-60fe0cc1.pth",
            first_conv="conv1.0",
        ),
        "resnet152s.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1s-dcc41b81.pth",
            first_conv="conv1.0",
        ),
        "resnext50_32x4d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext50_32x4d-e6a097c1.pth",
        ),
        "resnext101_32x4d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_32x4d-b253c8c4.pth",
        ),
        "resnext101_64x4d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_64x4d-f9a8e184.pth",
        ),
        "seresnext50_32x4d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext50_32x4d-90cf2d6e.pth",
        ),
        "seresnext101_32x4d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext101_32x4d-cf52900d.pth",
        ),
        "seresnext101_64x4d.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext101_64x4d-f9926f93.pth",
        ),
        "senet154.gluon_in1k": _gcfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_senet154-70a1a3c0.pth",
            first_conv="conv1.0",
        ),
    }
)


@register_model
def resnet10t(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-10-T model."""
    model_args = dict(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
    )
    return _create_resnet("resnet10t", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet14t(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-14-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=[1, 1, 1, 1],
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
    )
    return _create_resnet("resnet14t", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet18(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model."""
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return _create_resnet("resnet18", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet20_cifar(num_classes=10, down_type="conv", act="relu", **kwargs) -> ResNet:
    """Constructs a CIFAR ResNet-20 model."""
    return ResNetCIFAR(
        block=BasicBlockCIFAR,
        depth=20,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet20_cifar_preact(
    num_classes=10, down_type="conv", act="relu", **kwargs
) -> ResNet:
    """Constructs a PreAct CIFAR ResNet-20 model."""
    return ResNetCIFARPreAct(
        block=BasicBlockCIFARPreAct,
        depth=20,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def wide_resnet26_10_cifar(
    num_classes=10, down_type="conv", act="relu", **kwargs
) -> ResNet:
    """Constructs a CIFAR ResNet-26 model."""
    return ResNetCIFAR(
        block=BasicBlockCIFAR,
        depth=26,
        width_multiplier=10,
        # depth=14,
        # width_multiplier=1,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def wide_resnet26_10_cifar_ddu(num_classes=10, **kwargs) -> ResNet:
    """Constructs a CIFAR ResNet-26 model."""
    return WideResNet(
        spectral_normalization=True,
        mode=True,
        # depth=16,
        # widen_factor=1,
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        coeff=3,
    )


@register_model
def wide_resnet26_10_cifar_preact(
    num_classes=10, down_type="conv", act="relu", **kwargs
) -> ResNet:
    """Constructs a PreAct CIFAR ResNet-26 model."""
    return ResNetCIFARPreAct(
        block=BasicBlockCIFARPreAct,
        depth=26,
        width_multiplier=10,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet18d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-18-D model."""
    model_args = dict(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnet18d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet34(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return _create_resnet("resnet34", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet32_cifar(num_classes=10, down_type="conv", act="relu", **kwargs) -> ResNet:
    """Constructs a CIFAR ResNet-32 model."""
    return ResNetCIFAR(
        block=BasicBlockCIFAR,
        depth=32,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet32_cifar_preact(
    num_classes=10, down_type="conv", act="relu", **kwargs
) -> ResNet:
    """Constructs a PreAct CIFAR ResNet-32 model."""
    return ResNetCIFARPreAct(
        block=BasicBlockCIFARPreAct,
        depth=32,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet34d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model."""
    model_args = dict(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnet34d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-26 model."""
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2])
    return _create_resnet("resnet26", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26t(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
    )
    return _create_resnet("resnet26t", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnet26d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet("resnet50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50_cifar(num_classes=10, down_type="conv", act="relu", **kwargs) -> ResNet:
    """Constructs a CIFAR ResNet-50 model."""
    return ResNetCIFAR(
        block=BottleneckCIFAR,
        depth=50,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet50_cifar_preact(
    num_classes=10, down_type="conv", act="relu", **kwargs
) -> ResNet:
    """Constructs a PreAct CIFAR ResNet-50 model."""
    return ResNetCIFARPreAct(
        block=BottleneckCIFARPreAct,
        depth=50,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet50c(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-C model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type="deep"
    )
    return _create_resnet("resnet50c", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnet50d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-S model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=64, stem_type="deep"
    )
    return _create_resnet("resnet50s", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50t(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
    )
    return _create_resnet("resnet50t", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return _create_resnet("resnet101", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet110_cifar(num_classes=10, down_type="conv", act="relu", **kwargs) -> ResNet:
    """Constructs a CIFAR ResNet-110 model."""
    return ResNetCIFAR(
        block=BottleneckCIFAR,
        depth=110,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet110_cifar_preact(
    num_classes=10, down_type="conv", act="relu", **kwargs
) -> ResNet:
    """Constructs a PreAct CIFAR ResNet-110 model."""
    return ResNetCIFARPreAct(
        block=BottleneckCIFARPreAct,
        depth=110,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet101c(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-C model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type="deep"
    )
    return _create_resnet("resnet101c", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnet101d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-S model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=64, stem_type="deep"
    )
    return _create_resnet("resnet101s", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model."""
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3])
    return _create_resnet("resnet152", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152_cifar(num_classes=10, down_type="conv", act="relu", **kwargs) -> ResNet:
    """Constructs a CIFAR ResNet-152 model."""
    return ResNetCIFAR(
        block=BottleneckCIFAR,
        depth=152,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet152_cifar_preact(
    num_classes=10, down_type="conv", act="relu", **kwargs
) -> ResNet:
    """Constructs a PreAct CIFAR ResNet-152 model."""
    return ResNetCIFARPreAct(
        block=BottleneckCIFARPreAct,
        depth=152,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )


@register_model
def resnet152c(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-C model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type="deep"
    )
    return _create_resnet("resnet152c", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnet152d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-S model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=64, stem_type="deep"
    )
    return _create_resnet("resnet152s", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200 model."""
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3])
    return _create_resnet("resnet200", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 24, 36, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnet200d", pretrained, **dict(model_args, **kwargs))


@register_model
def wide_resnet50_2(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128)
    return _create_resnet("wide_resnet50_2", pretrained, **dict(model_args, **kwargs))


@register_model
def wide_resnet101_2(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], base_width=128)
    return _create_resnet("wide_resnet101_2", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50_gn(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model w/ GroupNorm"""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet("resnet50_gn", pretrained, norm_layer=GroupNorm, **model_args)


@register_model
def resnext50_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt50-32x4d model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4
    )
    return _create_resnet("resnext50_32x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext50d_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnext50d_32x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x4d model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4
    )
    return _create_resnet("resnext101_32x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x8d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x8d model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8
    )
    return _create_resnet("resnext101_32x8d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x16d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x16d model"""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=16
    )
    return _create_resnet("resnext101_32x16d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x32d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x32d model"""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=32
    )
    return _create_resnet("resnext101_32x32d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_64x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt101-64x4d model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4
    )
    return _create_resnet("resnext101_64x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet26t(pretrained=False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet("ecaresnet26t", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with eca."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet("ecaresnet50d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet50d_pruned(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model pruned with eca.
    The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet(
        "ecaresnet50d_pruned", pretrained, pruned=True, **dict(model_args, **kwargs)
    )


@register_model
def ecaresnet50t(pretrained=False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-50-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet("ecaresnet50t", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnetlight(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D light model with eca."""
    model_args = dict(
        block=Bottleneck,
        layers=[1, 1, 11, 3],
        stem_width=32,
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet("ecaresnetlight", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet101d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with eca."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet("ecaresnet101d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet101d_pruned(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model pruned with eca.
    The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet(
        "ecaresnet101d_pruned", pretrained, pruned=True, **dict(model_args, **kwargs)
    )


@register_model
def ecaresnet200d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model with ECA."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 24, 36, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet("ecaresnet200d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet269d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-269-D model with ECA."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 30, 48, 8],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet("ecaresnet269d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnext26t_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet(
        "ecaresnext26t_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def ecaresnext50t_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-50-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
        block_args=dict(attn_layer="eca"),
    )
    return _create_resnet(
        "ecaresnext50t_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnet18(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=BasicBlock, layers=[2, 2, 2, 2], block_args=dict(attn_layer="se")
    )
    return _create_resnet("seresnet18", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet34(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=BasicBlock, layers=[3, 4, 6, 3], block_args=dict(attn_layer="se")
    )
    return _create_resnet("seresnet34", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet50(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], block_args=dict(attn_layer="se")
    )
    return _create_resnet("seresnet50", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet50t(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet("seresnet50t", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet101(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], block_args=dict(attn_layer="se")
    )
    return _create_resnet("seresnet101", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet152(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], block_args=dict(attn_layer="se")
    )
    return _create_resnet("seresnet152", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet152d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet("seresnet152d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet200d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model with SE attn."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 24, 36, 3],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet("seresnet200d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet269d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-269-D model with SE attn."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 30, 48, 8],
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet("seresnet269d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext26d_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-26-D model.`
    This is technically a 28 layer ResNet, using the 'D' modifier from Gluon / bag-of-tricks for
    combination of deep stem and avg_pool in downsample.
    """
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet(
        "seresnext26d_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext26t_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem.
    """
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep_tiered",
        down_type="avg",
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet(
        "seresnext26t_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext26tn_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-26-T model.
    NOTE I deprecated previous 't' model defs and replaced 't' with 'tn', this was the only tn model of note
    so keeping this def for backwards compat with any uses out there. Old 't' model is lost.
    """
    return seresnext26t_32x4d(pretrained=pretrained, **kwargs)


@register_model
def seresnext50_32x4d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        cardinality=32,
        base_width=4,
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet("seresnext50_32x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101_32x4d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        cardinality=32,
        base_width=4,
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet(
        "seresnext101_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext101_32x8d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        cardinality=32,
        base_width=8,
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet(
        "seresnext101_32x8d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext101d_32x8d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        cardinality=32,
        base_width=8,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet(
        "seresnext101d_32x8d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext101_64x4d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        cardinality=64,
        base_width=4,
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet(
        "seresnext101_64x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def senet154(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        cardinality=64,
        base_width=4,
        stem_type="deep",
        down_kernel_size=3,
        block_reduce_first=2,
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet("senet154", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur18(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model with blur anti-aliasing"""
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], aa_layer=BlurPool2d)
    return _create_resnet("resnetblur18", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model with blur anti-aliasing"""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d)
    return _create_resnet("resnetblur50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with blur anti-aliasing"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        aa_layer=BlurPool2d,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnetblur50d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur101d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with blur anti-aliasing"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        aa_layer=BlurPool2d,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnetblur101d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa34d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model w/ avgpool anti-aliasing"""
    model_args = dict(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        aa_layer=nn.AvgPool2d,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnetaa34d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model with avgpool anti-aliasing"""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d)
    return _create_resnet("resnetaa50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with avgpool anti-aliasing"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        aa_layer=nn.AvgPool2d,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnetaa50d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa101d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with avgpool anti-aliasing"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        aa_layer=nn.AvgPool2d,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
    )
    return _create_resnet("resnetaa101d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnetaa50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE=ResNet-50-D model with avgpool anti-aliasing"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        aa_layer=nn.AvgPool2d,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet("seresnetaa50d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnextaa101d_32x8d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE=ResNeXt-101-D 32x8d model with avgpool anti-aliasing"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        cardinality=32,
        base_width=8,
        stem_width=32,
        stem_type="deep",
        down_type="avg",
        aa_layer=nn.AvgPool2d,
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet(
        "seresnextaa101d_32x8d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnextaa201d_32x8d(pretrained=False, **kwargs):
    """Constructs a SE=ResNeXt-101-D 32x8d model with avgpool anti-aliasing"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 24, 36, 4],
        cardinality=32,
        base_width=8,
        stem_width=64,
        stem_type="deep",
        down_type="avg",
        aa_layer=nn.AvgPool2d,
        block_args=dict(attn_layer="se"),
    )
    return _create_resnet(
        "seresnextaa201d_32x8d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def resnetrs50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-50 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn("se"), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        down_type="avg",
        block_args=dict(attn_layer=attn_layer),
    )
    return _create_resnet("resnetrs50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs101(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-101 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn("se"), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        down_type="avg",
        block_args=dict(attn_layer=attn_layer),
    )
    return _create_resnet("resnetrs101", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs152(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-152 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn("se"), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        down_type="avg",
        block_args=dict(attn_layer=attn_layer),
    )
    return _create_resnet("resnetrs152", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs200(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-200 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn("se"), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck,
        layers=[3, 24, 36, 3],
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        down_type="avg",
        block_args=dict(attn_layer=attn_layer),
    )
    return _create_resnet("resnetrs200", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs270(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-270 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn("se"), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck,
        layers=[4, 29, 53, 4],
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        down_type="avg",
        block_args=dict(attn_layer=attn_layer),
    )
    return _create_resnet("resnetrs270", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs350(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-350 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn("se"), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck,
        layers=[4, 36, 72, 4],
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        down_type="avg",
        block_args=dict(attn_layer=attn_layer),
    )
    return _create_resnet("resnetrs350", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs420(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-420 model
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn("se"), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck,
        layers=[4, 44, 87, 4],
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        down_type="avg",
        block_args=dict(attn_layer=attn_layer),
    )
    return _create_resnet("resnetrs420", pretrained, **dict(model_args, **kwargs))


register_model_deprecations(
    __name__,
    {
        "tv_resnet34": "resnet34.tv_in1k",
        "tv_resnet50": "resnet50.tv_in1k",
        "tv_resnet101": "resnet101.tv_in1k",
        "tv_resnet152": "resnet152.tv_in1k",
        "tv_resnext50_32x4d": "resnext50_32x4d.tv_in1k",
        "ig_resnext101_32x8d": "resnext101_32x8d.fb_wsl_ig1b_ft_in1k",
        "ig_resnext101_32x16d": "resnext101_32x8d.fb_wsl_ig1b_ft_in1k",
        "ig_resnext101_32x32d": "resnext101_32x8d.fb_wsl_ig1b_ft_in1k",
        "ig_resnext101_32x48d": "resnext101_32x8d.fb_wsl_ig1b_ft_in1k",
        "ssl_resnet18": "resnet18.fb_ssl_yfcc100m_ft_in1k",
        "ssl_resnet50": "resnet50.fb_ssl_yfcc100m_ft_in1k",
        "ssl_resnext50_32x4d": "resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k",
        "ssl_resnext101_32x4d": "resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k",
        "ssl_resnext101_32x8d": "resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k",
        "ssl_resnext101_32x16d": "resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k",
        "swsl_resnet18": "resnet18.fb_swsl_ig1b_ft_in1k",
        "swsl_resnet50": "resnet50.fb_swsl_ig1b_ft_in1k",
        "swsl_resnext50_32x4d": "resnext50_32x4d.fb_swsl_ig1b_ft_in1k",
        "swsl_resnext101_32x4d": "resnext101_32x4d.fb_swsl_ig1b_ft_in1k",
        "swsl_resnext101_32x8d": "resnext101_32x8d.fb_swsl_ig1b_ft_in1k",
        "swsl_resnext101_32x16d": "resnext101_32x16d.fb_swsl_ig1b_ft_in1k",
        "gluon_resnet18_v1b": "resnet18.gluon_in1k",
        "gluon_resnet34_v1b": "resnet34.gluon_in1k",
        "gluon_resnet50_v1b": "resnet50.gluon_in1k",
        "gluon_resnet101_v1b": "resnet101.gluon_in1k",
        "gluon_resnet152_v1b": "resnet152.gluon_in1k",
        "gluon_resnet50_v1c": "resnet50c.gluon_in1k",
        "gluon_resnet101_v1c": "resnet101c.gluon_in1k",
        "gluon_resnet152_v1c": "resnet152c.gluon_in1k",
        "gluon_resnet50_v1d": "resnet50d.gluon_in1k",
        "gluon_resnet101_v1d": "resnet101d.gluon_in1k",
        "gluon_resnet152_v1d": "resnet152d.gluon_in1k",
        "gluon_resnet50_v1s": "resnet50s.gluon_in1k",
        "gluon_resnet101_v1s": "resnet101s.gluon_in1k",
        "gluon_resnet152_v1s": "resnet152s.gluon_in1k",
        "gluon_resnext50_32x4d": "resnext50_32x4d.gluon_in1k",
        "gluon_resnext101_32x4d": "resnext101_32x4d.gluon_in1k",
        "gluon_resnext101_64x4d": "resnext101_64x4d.gluon_in1k",
        "gluon_seresnext50_32x4d": "seresnext50_32x4d.gluon_in1k",
        "gluon_seresnext101_32x4d": "seresnext101_32x4d.gluon_in1k",
        "gluon_seresnext101_64x4d": "seresnext101_64x4d.gluon_in1k",
        "gluon_senet154": "senet154.gluon_in1k",
    },
)
