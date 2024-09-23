from enum import Enum

from clinicadl.utils.enum import BaseEnum


class ClassificationLoss(str, BaseEnum):
    """Losses that can be used only for classification."""

    CROSS_ENTROPY = "CrossEntropyLoss"  # for multi-class classification, inputs are unormalized logits and targets are int (same dimension without the class channel)
    NLL = "NLLLoss"  # for multi-class classification, inputs are log-probabilities and targets are int (same dimension without the class channel)
    MULTI_MARGIN = "MultiMarginLoss"  # no particular restriction on the input, targets are int (same dimension without th class channel)
    BCE = "BCELoss"  # for binary classification, targets and inputs should be probabilities and have same shape
    BCE_LOGITS = "BCEWithLogitsLoss"  # for binary classification, targets should be probabilities and inputs logits, and have the same shape. More stable numerically


class ImplementedLoss(str, Enum):
    """Implemented losses in ClinicaDL."""

    CROSS_ENTROPY = "CrossEntropyLoss"
    NLL = "NLLLoss"
    MULTI_MARGIN = "MultiMarginLoss"
    BCE = "BCELoss"
    BCE_LOGITS = "BCEWithLogitsLoss"

    L1 = "L1Loss"
    MSE = "MSELoss"
    HUBER = "HuberLoss"
    SMOOTH_L1 = "SmoothL1Loss"
    KLDIV = "KLDivLoss"  # if log_target=False, target must be positive

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented losses are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class Reduction(str, Enum):
    """Supported reduction method in ClinicaDL."""

    MEAN = "mean"
    SUM = "sum"


class Order(int, Enum):
    """Supported order of L-norm for MultiMarginLoss."""

    ONE = 1
    TWO = 2
