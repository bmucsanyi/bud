import warnings

from ._builder import *
from ._helpers import *
from ._manipulate import *
from ._prune import *

warnings.warn(
    f"Importing from {__name__} is deprecated, please import via timm.models",
    DeprecationWarning,
)
