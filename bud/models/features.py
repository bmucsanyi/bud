import warnings

from ._features import *

warnings.warn(
    f"Importing from {__name__} is deprecated, please import via timm.models",
    DeprecationWarning,
)
