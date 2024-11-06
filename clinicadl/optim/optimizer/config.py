from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    computed_field,
    field_validator,
    model_validator,
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
    freeze: Optional[Union[str, List[str]]] = None
    weight_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    foreach: Union[
        Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    maximize: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES
    differentiable: Union[
        bool, Dict[str, bool], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )

    @computed_field
    @property
    @abstractmethod
    def name(self) -> ImplementedOptimizer:
        """The name of the optimizer."""

    @field_validator("freeze", mode="after")
    @classmethod
    def validator_freeze(cls, v):
        """To always have a list for 'freeze'."""
        if isinstance(v, str):
            return [v]
        return v

    @classmethod
    def validator_proba(cls, v, ctx):
        """To validate probabilities."""
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

    @field_validator("*", mode="after")
    @classmethod
    def check_else(cls, v, ctx):
        """Checks that 'ELSE' is always in dicts."""
        name = ctx.field_name
        if isinstance(v, dict) and "ELSE" not in v:
            raise ValueError(
                f"If you pass a dict to {name}, it must contain the key 'ELSE', that corresponds "
                f"to the value applied to the rest of the parameters. Got: {v}"
            )
        return v

    @model_validator(mode="after")
    def check_param_groups(self):
        """Check that a parameter group is not passed both in a field and in 'freeze'."""
        if self.freeze is not None:
            for field, value in self:
                if isinstance(value, dict):
                    for group in value:
                        if group in self.freeze:
                            raise ValueError(
                                f"You mentioned the parameter group {group} in {field}, but this parameter "
                                "group is also passed in 'freeze'."
                            )
        return self

    def get_all_groups(self) -> Set[str]:
        """
        Returns all groups mentioned by the user in the fields.

        Returns
        -------
        Set[str]
            the groups.
        """
        groups = set()
        for _, value in self:
            if isinstance(value, dict):
                groups.update(set(value.keys()))

        return groups


class _CapturableConfig(OptimizerConfig):
    """Base config class for optimizer with 'capturable' option."""

    capturable: Union[
        bool, Dict[str, bool], DefaultFromLibrary
    ] = DefaultFromLibrary.YES


class _FusedConfig(OptimizerConfig):
    """Base config class for optimizer with 'fused' option."""

    fused: Union[
        Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES


class _EpsConfig(OptimizerConfig):
    """Base config class for optimizer with 'eps' option."""

    eps: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES


class _MomentumConfig(OptimizerConfig):
    """Base config class for optimizer with 'eps' option."""

    momentum: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES


class AdadeltaConfig(_EpsConfig, _CapturableConfig):
    """Config class for Adadelta optimizer."""

    rho: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADADELTA

    @field_validator("rho")
    @classmethod
    def validator_rho(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class AdagradConfig(_EpsConfig, _FusedConfig):
    """Config class for Adagrad optimizer."""

    lr_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    initial_accumulator_value: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADAGRAD


class AdamConfig(_EpsConfig, _CapturableConfig, _FusedConfig):
    """Config class for Adam optimizer."""

    betas: Union[
        Tuple[NonNegativeFloat, NonNegativeFloat],
        Dict[str, Tuple[NonNegativeFloat, NonNegativeFloat]],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
    amsgrad: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.ADAM

    @field_validator("betas")
    @classmethod
    def validator_betas(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class RMSpropConfig(_EpsConfig, _CapturableConfig, _MomentumConfig):
    """Config class for RMSprop optimizer."""

    alpha: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    centered: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.RMS_PROP

    @field_validator("alpha")
    @classmethod
    def validator_alpha(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class SGDConfig(_FusedConfig, _MomentumConfig):
    """Config class for SGD optimizer."""

    dampening: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    nesterov: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedOptimizer:
        """The name of the optimizer."""
        return ImplementedOptimizer.SGD

    @field_validator("dampening")
    @classmethod
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
