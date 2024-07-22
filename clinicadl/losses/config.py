from typing import List, Optional

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat

from clinicadl.utils.enum import BaseEnum


class ClassificationLoss(str, BaseEnum):
    """Losses that can be used only for classification."""

    CROSS_ENTROPY = "CrossEntropyLoss"
    MULTI_MARGIN = "MultiMarginLoss"


class ImplementedLoss(str, BaseEnum):
    """Implemented losses in ClinicaDL."""

    CROSS_ENTROPY = "CrossEntropyLoss"
    MULTI_MARGIN = "MultiMarginLoss"
    L1 = "L1Loss"
    MSE = "MSELoss"
    HUBER = "HuberLoss"
    SMOOTH_L1 = "SmoothL1Loss"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented losses are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class Reduction(str, BaseEnum):
    """Supported reduction method in ClinicaDL."""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


class Order(int, BaseEnum):
    """Supported order of L-norm for MultiMarginLoss."""

    ONE = 1
    TWO = 2


class LossConfig(BaseModel):
    """Config class to configure the loss function."""

    loss: ImplementedLoss = ImplementedLoss.MSE
    reduction: Reduction = Reduction.MEAN
    delta: PositiveFloat = 1.0
    beta: PositiveFloat = 1.0
    p: Order = Order.ONE
    margin: float = 1.0
    weight: Optional[List[NonNegativeFloat]] = None
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )
