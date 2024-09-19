from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .enum import ImplementedLoss, Order, Reduction

__all__ = [
    "LossConfig",
    "NLLConfig",
    "CrossEntropyConfig",
    "BCEConfig",
    "BCEWithLogitsConfig",
    "MultiMarginConfig",
    "KLDivConfig",
    "HuberConfig",
    "SmoothL1Config",
    "L1Config",
    "MSEConfig",
    "create_loss_config",
]


class LossConfig(BaseModel, ABC):
    """Base config class for the loss function."""

    reduction: Union[Reduction, DefaultFromLibrary] = DefaultFromLibrary.YES
    weight: Union[
        Optional[List[NonNegativeFloat]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )

    @computed_field
    @property
    @abstractmethod
    def loss(self) -> str:
        """The name of the loss."""


class NLLConfig(LossConfig):
    """Config class for Negative Log Likelihood loss."""

    ignore_index: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "NLLLoss"

    @field_validator("ignore_index")
    @classmethod
    def validator_ignore_index(cls, v):
        if isinstance(v, int):
            assert (
                v == -100 or 0 <= v
            ), "ignore_index must be a positive int (or -100 when disabled)."
        return v


class CrossEntropyConfig(NLLConfig):
    """Config class for Cross Entropy loss."""

    label_smoothing: Union[
        NonNegativeFloat, DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "CrossEntropyLoss"

    @field_validator("label_smoothing")
    @classmethod
    def validator_label_smoothing(cls, v):
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"label_smoothing must be between 0 and 1 but it has been set to {v}."
        return v


class BCEConfig(LossConfig):
    """Config class for Binary Cross Entropy loss."""

    weight: Optional[List[NonNegativeFloat]] = None

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "BCELoss"

    @field_validator("weight")
    @classmethod
    def validator_weight(cls, v):
        if v is not None:
            raise ValueError(
                "Cannot use weight with BCEWithLogitsLoss. If you want more flexibility, please use API mode."
            )
        return v


class BCEWithLogitsConfig(BCEConfig):
    """Config class for Binary Cross Entropy With Logits loss."""

    pos_weight: Union[Optional[List[Any]], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "BCEWithLogitsLoss"

    @field_validator("pos_weight")
    @classmethod
    def validator_pos_weight(cls, v):
        if isinstance(v, list):
            check = cls._recursive_float_check(v)
            if not check:
                raise ValueError(
                    f"elements in pos_weight must be non-negative float, got: {v}"
                )
        return v

    @classmethod
    def _recursive_float_check(cls, item):
        if isinstance(item, list):
            return all(cls._recursive_float_check(i) for i in item)
        else:
            return (isinstance(item, float) or isinstance(item, int)) and item >= 0


class MultiMarginConfig(LossConfig):
    """Config class for Multi Margin loss."""

    p: Union[Order, DefaultFromLibrary] = DefaultFromLibrary.YES
    margin: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "MultiMarginLoss"


class KLDivConfig(LossConfig):
    """Config class for Kullback-Leibler Divergence loss."""

    log_target: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "KLDivLoss"


class HuberConfig(LossConfig):
    """Config class for Huber loss."""

    delta: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "HuberLoss"


class SmoothL1Config(LossConfig):
    """Config class for Smooth L1 loss."""

    beta: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "SmoothL1Loss"


class L1Config(LossConfig):
    """Config class for L1 loss."""

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "L1Loss"


class MSEConfig(LossConfig):
    """Config class for Mean Squared Error loss."""

    @computed_field
    @property
    def loss(self) -> str:
        """The name of the loss."""
        return "MSELoss"


def create_loss_config(
    loss: Union[str, ImplementedLoss],
) -> Type[LossConfig]:
    """
    A factory function to create a config class suited for the loss.

    Parameters
    ----------
    loss : Union[str, ImplementedLoss]
        The name of the loss.

    Returns
    -------
    Type[LossConfig]
        The config class.

    Raises
    ------
    ValueError
        If `loss` is not supported.
    """
    loss = ImplementedLoss(loss)
    config_name = "".join([loss.replace(" ", ""), "Config"])
    config = globals()[config_name]

    return config
