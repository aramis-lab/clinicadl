from typing import Dict, List, Set, Union

import torch.optim as optim
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    LRSchedulerConfig,
    _FactorConfig,
    _GammaConfig,
    _LastEpochConfig,
    _TotalItersConfig,
)
from .enum import ImplementedLRScheduler, Mode, ThresholdMode

__all__ = [
    "ConstantLRConfig",
    "ExponentialLRConfig",
    "LinearLRConfig",
    "StepLRConfig",
    "MultiStepLRConfig",
    "PolynomialLR",
    "ReduceLROnPlateauConfig",
]


class ConstantLRConfig(
    LRSchedulerConfig, _FactorConfig, _TotalItersConfig, _LastEpochConfig
):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.ConstantLR`.
    """

    def __init__(
        self,
        factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        total_iters: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            factor=factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the scheduler."""
        return ImplementedLRScheduler.CONSTANT.value

    def _get_class(self) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return optim.lr_scheduler.ConstantLR


class ExponentialLRConfig(LRSchedulerConfig, _GammaConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.ExponentialLR`.
    """

    def __init__(
        self,
        gamma: PositiveFloat,
        last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            gamma=gamma,
            last_epoch=last_epoch,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the scheduler."""
        return ImplementedLRScheduler.EXPONENTIAL.value

    def _get_class(self) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return optim.lr_scheduler.ExponentialLR


class LinearLRConfig(LRSchedulerConfig, _TotalItersConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.LinearLR`.
    """

    start_factor: PositiveFloat
    end_factor: PositiveFloat

    def __init__(
        self,
        start_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        end_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        total_iters: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the scheduler."""
        return ImplementedLRScheduler.LINEAR.value

    def _get_class(self) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return optim.lr_scheduler.LinearLR


class StepLRConfig(LRSchedulerConfig, _GammaConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.StepLR`.
    """

    step_size: PositiveInt

    def __init__(
        self,
        step_size: PositiveInt,
        gamma: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the scheduler."""
        return ImplementedLRScheduler.STEP.value

    def _get_class(self) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return optim.lr_scheduler.StepLR


class MultiStepLRConfig(LRSchedulerConfig, _GammaConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.MultiStepLR`.
    """

    milestones: List[PositiveInt]

    def __init__(
        self,
        milestones: List[PositiveInt],
        gamma: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            milestones=milestones,
            gamma=gamma,
            last_epoch=last_epoch,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the scheduler."""
        return ImplementedLRScheduler.MULTI_STEP.value

    def _get_class(self) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return optim.lr_scheduler.MultiStepLR

    @field_validator("milestones", mode="after")
    @classmethod
    def validator_milestones(cls, v):
        import numpy as np

        assert len(np.unique(v)) == len(v), "Epoch(s) in 'milestones' should be unique."
        return sorted(v)


class PolynomialLR(LRSchedulerConfig, _TotalItersConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.PolynomialLR`.
    """

    power: float

    def __init__(
        self,
        total_iters: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        power: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES,
        last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            total_iters=total_iters,
            power=power,
            last_epoch=last_epoch,
        )

    @computed_field
    @property
    def name(self) -> str:
        """The name of the scheduler."""
        return ImplementedLRScheduler.POLYNOMIAL.value

    def _get_class(self) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return optim.lr_scheduler.PolynomialLR


class ReduceLROnPlateauConfig(LRSchedulerConfig, _FactorConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.
    """

    mode: Mode
    patience: NonNegativeInt
    threshold: NonNegativeFloat
    threshold_mode: ThresholdMode
    cooldown: NonNegativeInt
    min_lr: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]]
    eps: NonNegativeFloat

    def __init__(
        self,
        mode: Union[Mode, DefaultFromLibrary] = DefaultFromLibrary.YES,
        factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        patience: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        threshold: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        threshold_mode: Union[
            ThresholdMode, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        cooldown: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        min_lr: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        eps: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )

    @property
    def name(self) -> str:
        """The name of the scheduler."""
        return ImplementedLRScheduler.PLATEAU.value

    def _get_class(self) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return optim.lr_scheduler.ReduceLROnPlateau

    @field_validator("min_lr", mode="after")
    @classmethod
    def min_lr_validator(cls, v):
        """Checks that 'ELSE' is always in 'min_lr' if it is a dict."""
        return cls.group_validator(v, field_name="min_lr")


class OneCycleLRConfig(LRSchedulerConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.OneCycleLR`.
    """

    mode: Mode
    patience: NonNegativeInt
    threshold: NonNegativeFloat
    threshold_mode: ThresholdMode
    cooldown: NonNegativeInt
    min_lr: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]]
    eps: NonNegativeFloat

    def __init__(
        self,
        mode: Union[Mode, DefaultFromLibrary] = DefaultFromLibrary.YES,
        factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        patience: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        threshold: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
        threshold_mode: Union[
            ThresholdMode, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        cooldown: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        min_lr: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        eps: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )

    @property
    def name(self) -> str:
        """The name of the scheduler."""
        return ImplementedLRScheduler.ONE_CYCLE.value

    def _get_class(self) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return optim.lr_scheduler.OneCycleLR

    @field_validator("min_lr", mode="after")
    @classmethod
    def min_lr_validator(cls, v):
        """Checks that 'ELSE' is always in 'min_lr' if it is a dict."""
        return cls.group_validator(v, field_name="min_lr")
