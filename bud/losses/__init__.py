from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .binary_cross_entropy_loss import BinaryCrossEntropyLoss
from .bma_cross_entropy_loss import BMACrossEntropyLoss
from .correctness_prediction_loss import CorrectnessPredictionLoss
from .cross_entropy_loss import (
    LabelSmoothingCrossEntropyLoss,
    SoftTargetCrossEntropyLoss,
)
from .duq_loss import DUQLoss
from .fbar_cross_entropy_loss import FBarCrossEntropyLoss
from .jsd_loss import JsdCrossEntropyLoss
from .mcinfonce_loss import MCInfoNCELoss
from .nivmf_loss import NonIsotropicVMFLoss
from .loss_prediction_loss import LossPredictionLoss
from .edl_loss import EDLLoss
