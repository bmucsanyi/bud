# NOTE bud.models.layers is DEPRECATED, please use bud.layers, this is here to reduce breakages in transition
import warnings

from bud.layers.activations import *
from bud.layers.adaptive_avgmax_pool import (
    AdaptiveAvgMaxPool2d,
    SelectAdaptivePool2d,
    adaptive_avgmax_pool2d,
    select_adaptive_pool2d,
)
from bud.layers.attention_pool2d import (
    AttentionPool2d,
    RotaryEmbedding,
    RotAttentionPool2d,
)
from bud.layers.blur_pool import BlurPool2d
from bud.layers.classifier import ClassifierHead, create_classifier
from bud.layers.cond_conv2d import CondConv2d, get_condconv_initializer
from bud.layers.config import (
    is_exportable,
    is_no_jit,
    is_scriptable,
    set_exportable,
    set_layer_config,
    set_no_jit,
    set_scriptable,
)
from bud.layers.conv2d_same import Conv2dSame, conv2d_same
from bud.layers.conv_bn_act import ConvBnAct, ConvNormAct, ConvNormActAa
from bud.layers.create_act import create_act_layer, get_act_fn, get_act_layer
from bud.layers.create_attn import create_attn, get_attn
from bud.layers.create_conv2d import create_conv2d
from bud.layers.create_norm import create_norm_layer, get_norm_layer
from bud.layers.create_norm_act import create_norm_act_layer, get_norm_act_layer
from bud.layers.drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from bud.layers.eca import (
    CecaModule,
    CircularEfficientChannelAttn,
    EcaModule,
    EfficientChannelAttn,
)
from bud.layers.evo_norm import (
    EvoNorm2dB0,
    EvoNorm2dB1,
    EvoNorm2dB2,
    EvoNorm2dS0,
    EvoNorm2dS0a,
    EvoNorm2dS1,
    EvoNorm2dS1a,
    EvoNorm2dS2,
    EvoNorm2dS2a,
)
from bud.layers.fast_norm import (
    fast_group_norm,
    fast_layer_norm,
    is_fast_norm,
    set_fast_norm,
)
from bud.layers.filter_response_norm import (
    FilterResponseNormAct2d,
    FilterResponseNormTlu2d,
)
from bud.layers.gather_excite import GatherExcite
from bud.layers.global_context import GlobalContext
from bud.layers.helpers import (
    extend_tuple,
    make_divisible,
    to_2tuple,
    to_3tuple,
    to_4tuple,
    to_ntuple,
)
from bud.layers.inplace_abn import InplaceAbn
from bud.layers.linear import Linear
from bud.layers.mixed_conv2d import MixedConv2d
from bud.layers.mlp import ConvMlp, GatedMlp, GluMlp, Mlp
from bud.layers.non_local_attn import BatNonLocalAttn, NonLocalAttn
from bud.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d
from bud.layers.norm_act import BatchNormAct2d, GroupNormAct, convert_sync_batchnorm
from bud.layers.padding import get_padding, get_same_padding, pad_same
from bud.layers.patch_embed import PatchEmbed
from bud.layers.pool2d_same import AvgPool2dSame, create_pool2d
from bud.layers.selective_kernel import SelectiveKernel
from bud.layers.separable_conv import SeparableConv2d, SeparableConvNormAct
from bud.layers.space_to_depth import SpaceToDepthModule
from bud.layers.split_attn import SplitAttn
from bud.layers.split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from bud.layers.squeeze_excite import (
    EffectiveSEModule,
    EffectiveSqueezeExcite,
    SEModule,
    SqueezeExcite,
)
from bud.layers.std_conv import (
    ScaledStdConv2d,
    ScaledStdConv2dSame,
    StdConv2d,
    StdConv2dSame,
)
from bud.layers.test_time_pool import TestTimePoolHead, apply_test_time_pool
from bud.layers.trace_utils import _assert, _float_to_int
from bud.layers.weight_init import (
    lecun_normal_,
    trunc_normal_,
    trunc_normal_tf_,
    variance_scaling_,
)

warnings.warn(
    f"Importing from {__name__} is deprecated, please import via bud.layers",
    DeprecationWarning,
)
