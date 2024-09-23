from enum import Enum
from typing import List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
    model_validator,
)

from clinicadl.utils.enum import BaseEnum
from clinicadl.utils.factories import DefaultFromLibrary


class ClassificationLoss(str, BaseEnum):
    """Losses that can be used only for classification."""

    CROSS_ENTROPY = "CrossEntropyLoss"  # for multi-class classification, inputs are unormalized logits and targets are int (same dimension without the class channel)
    MULTI_MARGIN = "MultiMarginLoss"  # no particular restriction on the input, targets are int (same dimension without th class channel)
    BCE = "BCELoss"  # for binary classification, targets and inputs should be probabilities and have same shape
    BCE_LOGITS = "BCEWithLogitsLoss"  # for binary classification, targets should be probabilities and inputs logits, and have the same shape. More stable numerically


class ImplementedLoss(str, Enum):
    """Implemented losses in ClinicaDL."""

    CROSS_ENTROPY = "CrossEntropyLoss"
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
    ] = DefaultFromLibrary.YES  # a weight for each class
    ignore_index: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES
    label_smoothing: Union[
        NonNegativeFloat, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    log_target: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    pos_weight: Union[
        Optional[List[NonNegativeFloat]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES  # a positive weight for each class
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )

    @field_validator("label_smoothing")
    @classmethod
    def validator_label_smoothing(cls, v):
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"label_smoothing must be between 0 and 1 but it has been set to {v}."
        return v

    @field_validator("ignore_index")
    @classmethod
    def validator_ignore_index(cls, v):
        if isinstance(v, int):
            assert (
                v == -100 or 0 <= v
            ), "ignore_index must be a positive int (or -100 when disabled)."
        return v

    @model_validator(mode="after")
    def model_validator(self):
        if (
            self.loss == ImplementedLoss.BCE_LOGITS
            and self.weight is not None
            and self.weight != DefaultFromLibrary.YES
        ):
            raise ValueError("Cannot use weight with BCEWithLogitsLoss.")
        elif (
            self.loss == ImplementedLoss.BCE
            and self.weight is not None
            and self.weight != DefaultFromLibrary.YES
        ):
            raise ValueError("Cannot use weight with BCELoss.")
