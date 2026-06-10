from enum import Enum


class ImplementedLRScheduler(str, Enum):
    """Implemented LR schedulers in ClinicaDL."""

    CONSTANT = "ConstantLR"
    EXPONENTIAL = "ExponentialLR"
    LINEAR = "LinearLR"
    STEP = "StepLR"
    MULTI_STEP = "MultiStepLR"
    PLATEAU = "ReduceLROnPlateau"
    POLYNOMIAL = "PolynomialLR"
    ONE_CYCLE = "OneCycleLR"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented LR schedulers are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class LRSchedulerType(str, Enum):
    """Possible types of LR scheduler."""

    STEP = "step-based"
    EPOCH = "epoch-based"
    METRIC = "metric-based"


class Mode(str, Enum):
    """Supported mode for ReduceLROnPlateau."""

    MIN = "min"
    MAX = "max"


class ThresholdMode(str, Enum):
    """Supported threshold mode for ReduceLROnPlateau."""

    ABS = "abs"
    REL = "rel"


class AnnealingStrategy(str, Enum):
    """Supported annealing strategy for OneCycleLR."""

    COS = "cos"
    LINEAR = "linear"
