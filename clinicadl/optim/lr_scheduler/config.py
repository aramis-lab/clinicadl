from abc import ABC, abstractmethod
from typing import Dict, List, Set, Type, Union

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

from .enum import ImplementedLRScheduler, Mode, ThresholdMode

__all__ = [
    "LRSchedulerConfig",
    "ConstantLRConfig",
    "LinearLRConfig",
    "StepLRConfig",
    "MultiStepLRConfig",
    "ReduceLROnPlateauConfig",
    "create_lr_scheduler_config",
]


class LRSchedulerConfig(ClinicaDLConfig, ABC):
    """Base config class for the LR scheduler."""

    @computed_field
    @property
    @abstractmethod
    def name(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""

    def get_all_groups(self) -> Set[str]:
        """
        Returns all  parameter groups mentioned by the user in the fields.
        For most schedulers, no group can be mentioned.
        """
        return set()


class _GammaConfig(ClinicaDLConfig):
    """Base config class for 'gamma' parameter."""

    gamma: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES


class _FactorConfig(ClinicaDLConfig):
    """Base config class for 'factor' parameter."""

    factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES


class _TotalItersConfig(ClinicaDLConfig):
    """Base config class for 'total_iters' parameter."""

    total_iters: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES


class _LastEpochConfig(ClinicaDLConfig):
    """Base config class for 'last_epoch' parameter."""

    last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES

    @field_validator("last_epoch")
    @classmethod
    def validator_last_epoch(cls, v):
        if isinstance(v, int):
            assert (
                -1 <= v
            ), f"last_epoch must be -1 or a non-negative int but it has been set to {v}."
        return v


class ConstantLRConfig(
    LRSchedulerConfig, _FactorConfig, _TotalItersConfig, _LastEpochConfig
):
    """Config class for ConstantLR scheduler."""

    @computed_field
    @property
    def name(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.CONSTANT


class LinearLRConfig(LRSchedulerConfig, _TotalItersConfig, _LastEpochConfig):
    """Config class for LinearLR scheduler."""

    start_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    end_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.LINEAR


class StepLRConfig(LRSchedulerConfig, _GammaConfig, _LastEpochConfig):
    """Config class for StepLR scheduler."""

    step_size: PositiveInt

    @computed_field
    @property
    def name(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.STEP


class MultiStepLRConfig(LRSchedulerConfig, _GammaConfig, _LastEpochConfig):
    """Config class for MultiStepLR scheduler."""

    milestones: List[PositiveInt]

    @computed_field
    @property
    def name(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.MULTI_STEP

    @field_validator("milestones", mode="after")
    @classmethod
    def validator_milestones(cls, v):
        import numpy as np

        assert len(np.unique(v)) == len(v), "Epoch(s) in 'milestones' should be unique."
        return sorted(v)


class ReduceLROnPlateauConfig(LRSchedulerConfig, _FactorConfig):
    """Config class for ReduceLROnPlateau scheduler."""

    mode: Union[Mode, DefaultFromLibrary] = DefaultFromLibrary.YES
    patience: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    threshold: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    threshold_mode: Union[ThresholdMode, DefaultFromLibrary] = DefaultFromLibrary.YES
    cooldown: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    min_lr: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    eps: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES

    @property
    def name(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.PLATEAU

    @field_validator("min_lr", mode="after")
    @classmethod
    def min_lr_validator(cls, v):
        """Checks that 'ELSE' is always in 'min_lr' if it is a dict."""
        if isinstance(v, dict) and "ELSE" not in v:
            raise ValueError(
                f"If you pass a dict to min_lr, it must contain the key 'ELSE', that corresponds "
                f"to the value applied to the rest of the parameters. Got: {v}"
            )
        return v

    def get_all_groups(self) -> Set[str]:
        """
        Returns all parameter groups mentioned by the user in the fields.

        Returns
        -------
        Set[str]
            the groups.
        """
        if isinstance(self.min_lr, dict):
            return set(self.min_lr.keys())
        else:
            return set()


def create_lr_scheduler_config(
    scheduler: Union[str, ImplementedLRScheduler],
) -> Type[LRSchedulerConfig]:
    """
    A factory function to create a config class suited for the LR scheduler.

    Parameters
    ----------
    scheduler : Union[str, ImplementedLRScheduler]
        The name of the LR scheduler.

    Returns
    -------
    Type[LRSchedulerConfig]
        The config class.

    Raises
    ------
    ValueError
        If `scheduler` is not supported.
    """
    scheduler = ImplementedLRScheduler(scheduler)
    config_name = "".join([scheduler, "Config"])
    config = globals()[config_name]

    return config
