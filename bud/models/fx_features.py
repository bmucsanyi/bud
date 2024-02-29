import warnings

from ._features_fx import *

warnings.warn(
    f"Importing from {__name__} is deprecated, please import via timm.models",
    DeprecationWarning,
)
