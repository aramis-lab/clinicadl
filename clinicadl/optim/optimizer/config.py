from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .enum import ImplementedOptimizer

__all__ = [
    "OptimizerConfig",
    "AdadeltaConfig",
    "AdagradConfig",
    "AdamConfig",
    "RMSpropConfig",
    "SGDConfig",
    "create_optimizer_config",
]


class OptimizerConfig(BaseModel, ABC):
    """Base config class for the optimizer."""

    lr: Union[
        PositiveFloat, Dict[str, PositiveFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    weight_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    eps: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    foreach: Union[
        Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    maximize: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES
    differentiable: Union[
        bool, Dict[str, bool], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    capturable: Union[
        bool, Dict[str, bool], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    fused: Union[
        Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )

    @computed_field
    @property
    @abstractmethod
    def optimizer(self) -> ImplementedOptimizer:
        """The name of the optimizer."""

    @classmethod
    def validator_proba(cls, v, ctx):
        name = ctx.field_name
        if isinstance(v, dict):
            for _, value in v.items():
                cls._validate_single_proba(value, name)
        else:
            cls._validate_single_proba(v, name)
        return v

    @staticmethod
    def _validate_single_proba(v, name):
        if isinstance(v, tuple):
            assert (
                0 <= v[0] <= 1
            ), f"{name} must be between 0 and 1 but it has been set to {v}."
            assert (
                0 <= v[1] <= 1
            ), f"{name} must be between 0 and 1 but it has been set to {v}."
        elif isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"{name} must be between 0 and 1 but it has been set to {v}."

    def get_all_groups(self) -> List[str]:
        """
        Returns all groups mentioned by the user in the fields.

        Returns
        -------
        List[str]
            The list of groups.
        """
        groups = set()
        for _, value in self.model_dump().items():
            if isinstance(value, dict):
                groups.update(set(value.keys()))

        return list(groups)


class AdadeltaConfig(OptimizerConfig):
    """Config class for Adadelta optimizer."""

    rho: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def optimizer(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADADELTA

    @field_validator("rho")
    def validator_rho(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class AdagradConfig(OptimizerConfig):
    """Config class for Adagrad optimizer."""

    lr_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    initial_accumulator_value: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def optimizer(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADAGRAD


class AdamConfig(OptimizerConfig):
    """Config class for Adam optimizer."""

    betas: Union[
        Tuple[NonNegativeFloat, NonNegativeFloat],
        Dict[str, Tuple[NonNegativeFloat, NonNegativeFloat]],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
    amsgrad: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def optimizer(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADAM

    @field_validator("betas")
    def validator_betas(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class RMSpropConfig(OptimizerConfig):
    """Config class for RMSprop optimizer."""

    alpha: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    momentum: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    centered: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def optimizer(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.RMS_PROP

    @field_validator("alpha")
    def validator_alpha(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class SGDConfig(OptimizerConfig):
    """Config class for SGD optimizer."""

    momentum: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    dampening: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    nesterov: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def optimizer(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.SGD

    @field_validator("dampening")
    def validator_dampening(cls, v, ctx):
        return cls.validator_proba(v, ctx)


def create_optimizer_config(
    optimizer: Union[str, ImplementedOptimizer],
) -> Type[OptimizerConfig]:
    """
    A factory function to create a config class suited for the optimizer.

    Parameters
    ----------
    optimizer : Union[str, ImplementedOptimizer]
        The name of the optimizer.

    Returns
    -------
    Type[OptimizerConfig]
        The config class.

    Raises
    ------
    ValueError
        If `optimizer` is not supported.
    """
    optimizer = ImplementedOptimizer(optimizer)
    config_name = "".join([optimizer, "Config"])
    config = globals()[config_name]

    return config
