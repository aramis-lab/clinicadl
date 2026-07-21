from enum import Enum


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

    DICE = "DiceLoss"
    DICE_CE = "DiceCELoss"
    DICE_FOCAL = "DiceFocalLoss"
    GENERALIZED_DICE = "GeneralizedDiceLoss"
    GENERALIZED_DICE_FOCAL = "GeneralizedDiceFocalLoss"
    FOCAL = "FocalLoss"
    TVERSKY = "TverskyLoss"
    SOFT_DICE = "SoftclDiceLoss"

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


class GeneralizedDiceWeight(str, Enum):
    """Supported class weighting modes for generalized Dice losses."""

    SQUARE = "square"
    SIMPLE = "simple"
    UNIFORM = "uniform"
