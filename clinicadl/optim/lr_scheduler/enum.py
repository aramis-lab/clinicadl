from enum import Enum


class ImplementedLRScheduler(str, Enum):
    """Implemented LR schedulers in ClinicaDL."""

    CONSTANT = "ConstantLR"
    LINEAR = "LinearLR"
    STEP = "StepLR"
    MULTI_STEP = "MultiStepLR"
    PLATEAU = "ReduceLROnPlateau"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented LR schedulers are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class Mode(str, Enum):
    """Supported mode for ReduceLROnPlateau."""

    MIN = "min"
    MAX = "max"


class ThresholdMode(str, Enum):
    """Supported threshold mode for ReduceLROnPlateau."""

    ABS = "abs"
    REL = "rel"
