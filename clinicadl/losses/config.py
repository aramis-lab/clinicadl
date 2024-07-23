from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat

from clinicadl.utils.enum import BaseEnum
from clinicadl.utils.factories import DefaultFromLibrary


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


class Reduction(str, Enum):
    """Supported reduction method in ClinicaDL."""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


class Order(int, Enum):
    """Supported order of L-norm for MultiMarginLoss."""

    ONE = 1
    TWO = 2


class LossConfig(BaseModel):
    """Config class to configure the loss function."""

    loss: ImplementedLoss = ImplementedLoss.MSE
    reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES
    delta: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    beta: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    p: Union[Order, DefaultFromLibrary] = DefaultFromLibrary.YES
    margin: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES
    weight: Union[
        Optional[List[NonNegativeFloat]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )
