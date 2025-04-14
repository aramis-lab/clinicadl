from enum import Enum

from clinicadl.utils.enum import BaseEnum


class ClassificationLoss(str, BaseEnum):
    """Losses that can be used only for classification."""

    CROSS_ENTROPY = "CrossEntropyLoss"
    NLL = "NLLLoss"
    MULTI_MARGIN = "MultiMarginLoss"
    BCE = "BCELoss"
    BCE_LOGITS = "BCEWithLogitsLoss"


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
    KLDIV = "KLDivLoss"

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
