from typing import Dict, List, Optional, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

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


class LRSchedulerConfig(BaseModel):
    """Base config class for the LR scheduler."""

    gamma: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    total_iters: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )

    @computed_field
    @property
    def scheduler(self) -> Optional[ImplementedLRScheduler]:
        """The name of the scheduler."""
        return None

    @field_validator("last_epoch")
    @classmethod
    def validator_last_epoch(cls, v):
        if isinstance(v, int):
            assert (
                -1 <= v
            ), f"last_epoch must be -1 or a non-negative int but it has been set to {v}."
        return v


class ConstantLRConfig(LRSchedulerConfig):
    """Config class for ConstantLR scheduler."""

    @computed_field
    @property
    def scheduler(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.CONSTANT


class LinearLRConfig(LRSchedulerConfig):
    """Config class for LinearLR scheduler."""

    start_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    end_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def scheduler(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.LINEAR


class StepLRConfig(LRSchedulerConfig):
    """Config class for StepLR scheduler."""

    step_size: PositiveInt

    @computed_field
    @property
    def scheduler(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.STEP


class MultiStepLRConfig(LRSchedulerConfig):
    """Config class for MultiStepLR scheduler."""

    milestones: List[PositiveInt]

    @computed_field
    @property
    def scheduler(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.MULTI_STEP

    @field_validator("milestones", mode="after")
    @classmethod
    def validator_milestones(cls, v):
        import numpy as np

        assert len(np.unique(v)) == len(v), "Epoch(s) in milestones should be unique."
        return sorted(v)


class ReduceLROnPlateauConfig(LRSchedulerConfig):
    """Config class for ReduceLROnPlateau scheduler."""

    mode: Union[Mode, DefaultFromLibrary] = DefaultFromLibrary.YES
    patience: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    threshold: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    threshold_mode: Union[ThresholdMode, DefaultFromLibrary] = DefaultFromLibrary.YES
    cooldown: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    min_lr: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    eps: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES

    @property
    def scheduler(self) -> ImplementedLRScheduler:
        """The name of the scheduler."""
        return ImplementedLRScheduler.PLATEAU


def create_lr_scheduler_config(
    scheduler: Optional[Union[str, ImplementedLRScheduler]],
) -> Type[LRSchedulerConfig]:
    """
    A factory function to create a config class suited for the LR scheduler.

    Parameters
    ----------
    scheduler : Optional[Union[str, ImplementedLRScheduler]]
        The name of the LR scheduler.
        Can be None if no LR scheduler will be used.

    Returns
    -------
    Type[LRSchedulerConfig]
        The config class.

    Raises
    ------
    ValueError
        If `scheduler` is not supported.
    """
    if scheduler is None:
        return LRSchedulerConfig

    scheduler = ImplementedLRScheduler(scheduler)
    config_name = "".join([scheduler, "Config"])
    config = globals()[config_name]

    return config
