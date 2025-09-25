from typing import Dict, List, Optional, Sequence, Union

import torch
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from torch.optim import Optimizer

from clinicadl.utils.factories import get_defaults_from

from .base import (
    LRSchedulerConfig,
    _LastEpochConfig,
)
from .enum import AnnealingStrategy, LRSchedulerType, Mode, ThresholdMode

__all__ = [
    "ConstantLRConfig",
    "ExponentialLRConfig",
    "LinearLRConfig",
    "StepLRConfig",
    "MultiStepLRConfig",
    "PolynomialLRConfig",
    "ReduceLROnPlateauConfig",
    "OneCycleLRConfig",
]

CONSTANT_LR_DEFAULTS = get_defaults_from(torch.optim.lr_scheduler.ConstantLR)
EXPO_LR_DEFAULTS = get_defaults_from(torch.optim.lr_scheduler.ExponentialLR)
LINEAR_LR_DEFAULTS = get_defaults_from(torch.optim.lr_scheduler.LinearLR)
STEP_LR_DEFAULTS = get_defaults_from(torch.optim.lr_scheduler.StepLR)
MULTI_STEP_LR_DEFAULTS = get_defaults_from(torch.optim.lr_scheduler.MultiStepLR)
POLY_LR_DEFAULTS = get_defaults_from(torch.optim.lr_scheduler.PolynomialLR)
REDUCE_LR_ON_PLATEAU_DEFAULTS = get_defaults_from(
    torch.optim.lr_scheduler.ReduceLROnPlateau
)
ONE_CYCLE_LR_DEFAULTS = get_defaults_from(torch.optim.lr_scheduler.OneCycleLR)


class ConstantLRConfig(LRSchedulerConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.ConstantLR`.
    """

    factor: PositiveFloat = CONSTANT_LR_DEFAULTS["factor"]
    total_iters: PositiveInt = CONSTANT_LR_DEFAULTS["total_iters"]
    last_epoch: int = CONSTANT_LR_DEFAULTS["last_epoch"]

    @classmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""
        return LRSchedulerType.EPOCH


class ExponentialLRConfig(LRSchedulerConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.ExponentialLR`.
    """

    gamma: PositiveFloat
    last_epoch: int = EXPO_LR_DEFAULTS["last_epoch"]

    @classmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""
        return LRSchedulerType.EPOCH


class LinearLRConfig(LRSchedulerConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.LinearLR`.
    """

    start_factor: PositiveFloat = LINEAR_LR_DEFAULTS["start_factor"]
    end_factor: PositiveFloat = LINEAR_LR_DEFAULTS["end_factor"]
    total_iters: PositiveInt = LINEAR_LR_DEFAULTS["total_iters"]
    last_epoch: int = LINEAR_LR_DEFAULTS["last_epoch"]

    @classmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""
        return LRSchedulerType.EPOCH


class StepLRConfig(LRSchedulerConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.StepLR`.
    """

    step_size: PositiveInt
    gamma: PositiveFloat = STEP_LR_DEFAULTS["gamma"]
    last_epoch: int = STEP_LR_DEFAULTS["last_epoch"]

    @classmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""
        return LRSchedulerType.EPOCH


class MultiStepLRConfig(LRSchedulerConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.MultiStepLR`.
    """

    milestones: List[PositiveInt]
    gamma: PositiveFloat = MULTI_STEP_LR_DEFAULTS["gamma"]
    last_epoch: int = MULTI_STEP_LR_DEFAULTS["last_epoch"]

    @field_validator("milestones", mode="after")
    @classmethod
    def validator_milestones(cls, v):
        import numpy as np

        assert len(np.unique(v)) == len(v), "Epoch(s) in 'milestones' should be unique."
        return sorted(v)

    @classmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""
        return LRSchedulerType.EPOCH


class PolynomialLRConfig(LRSchedulerConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.PolynomialLR`.
    """

    total_iters: PositiveInt = POLY_LR_DEFAULTS["total_iters"]
    power: float = POLY_LR_DEFAULTS["power"]
    last_epoch: int = POLY_LR_DEFAULTS["last_epoch"]

    @classmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""
        return LRSchedulerType.EPOCH


class ReduceLROnPlateauConfig(LRSchedulerConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.
    """

    mode: Mode = REDUCE_LR_ON_PLATEAU_DEFAULTS["mode"]
    factor: PositiveFloat = REDUCE_LR_ON_PLATEAU_DEFAULTS["factor"]
    patience: NonNegativeInt = REDUCE_LR_ON_PLATEAU_DEFAULTS["patience"]
    threshold: NonNegativeFloat = REDUCE_LR_ON_PLATEAU_DEFAULTS["threshold"]
    threshold_mode: ThresholdMode = REDUCE_LR_ON_PLATEAU_DEFAULTS["threshold_mode"]
    cooldown: NonNegativeInt = REDUCE_LR_ON_PLATEAU_DEFAULTS["cooldown"]
    min_lr: Union[
        NonNegativeFloat, Sequence[NonNegativeFloat], Dict[str, NonNegativeFloat]
    ] = REDUCE_LR_ON_PLATEAU_DEFAULTS["min_lr"]
    eps: NonNegativeFloat = REDUCE_LR_ON_PLATEAU_DEFAULTS["eps"]

    @field_validator("min_lr", mode="after")
    @classmethod
    def min_lr_validator(cls, v):
        """Checks that 'ELSE' is always in 'min_lr' if it is a dict."""
        return cls.group_validator(v, field_name="min_lr")

    @classmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""
        return LRSchedulerType.METRIC


class OneCycleLRConfig(LRSchedulerConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.OneCycleLR`.
    """

    max_lr: Union[PositiveFloat, Sequence[PositiveFloat], Dict[str, PositiveFloat]]
    total_steps: Optional[PositiveInt] = ONE_CYCLE_LR_DEFAULTS["total_steps"]
    epochs: Optional[PositiveInt] = ONE_CYCLE_LR_DEFAULTS["epochs"]
    steps_per_epoch: Optional[PositiveInt] = ONE_CYCLE_LR_DEFAULTS["steps_per_epoch"]
    pct_start: NonNegativeFloat = ONE_CYCLE_LR_DEFAULTS["pct_start"]
    anneal_strategy: AnnealingStrategy = ONE_CYCLE_LR_DEFAULTS["anneal_strategy"]
    cycle_momentum: bool = ONE_CYCLE_LR_DEFAULTS["cycle_momentum"]
    base_momentum: Union[
        NonNegativeFloat, Sequence[NonNegativeFloat], Dict[str, NonNegativeFloat]
    ] = ONE_CYCLE_LR_DEFAULTS["base_momentum"]
    max_momentum: Union[
        NonNegativeFloat, Sequence[NonNegativeFloat], Dict[str, NonNegativeFloat]
    ] = ONE_CYCLE_LR_DEFAULTS["max_momentum"]
    div_factor: PositiveFloat = ONE_CYCLE_LR_DEFAULTS["div_factor"]
    final_div_factor: PositiveFloat = ONE_CYCLE_LR_DEFAULTS["final_div_factor"]
    three_phase: bool = ONE_CYCLE_LR_DEFAULTS["three_phase"]
    last_epoch: int = ONE_CYCLE_LR_DEFAULTS["last_epoch"]

    @model_validator(mode="after")
    def check_n_steps(self):
        """
        Checks that either 'total_steps' is passed, or both 'epochs' AND 'steps_per_epoch'.
        """
        if self.total_steps and (self.epochs or self.steps_per_epoch):
            raise ValueError(
                "You can't pass 'epochs' or 'steps_per_epoch' if you pass 'total_steps'. "
                f"Got total_steps={self.total_steps}, epochs={self.epochs} "
                f"and steps_per_epoch={self.steps_per_epoch}."
            )
        elif not self.total_steps and not (self.epochs and self.steps_per_epoch):
            raise ValueError(
                "If you don't pass 'total_steps', you must pass 'epochs' AND 'steps_per_epoch'. "
                f"Got total_steps={self.total_steps}, epochs={self.epochs} "
                f"and steps_per_epoch={self.steps_per_epoch}."
            )
        return self

    @field_validator("pct_start", mode="after")
    @classmethod
    def validator_proba(cls, v):
        """Checks that 'pct_start' is a probability."""
        if not 0 < v < 1:
            raise ValueError(f"'pct_start' must be between 0 and 1 (strictly). Got {v}")
        return v

    @field_validator("max_lr", "base_momentum", "max_momentum", mode="after")
    @classmethod
    def parameter_group_validator(cls, v, ctx):
        """Checks that 'ELSE' is always in a field if it is a dict."""
        return cls.group_validator(v, field_name=ctx.field_name)

    def _check_optimizer_consistency(self, optimizer: Optimizer) -> None:
        """
        Checks if LR scheduler and optimizers are consistent.
        """
        super()._check_optimizer_consistency(optimizer)
        if self.cycle_momentum:
            if (
                "momentum" not in optimizer.defaults
                and "betas" not in optimizer.defaults
            ):
                raise ValueError(
                    "If 'cycle_momentum' is True in OneCycleLR, the optimizer requires a momentum."
                )

    @classmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""
        return LRSchedulerType.STEP
