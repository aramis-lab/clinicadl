from typing import Dict, List, Optional, Union

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import get_defaults_from

from .base import (
    LRSchedulerConfig,
    _FactorConfig,
    _GammaConfig,
    _LastEpochConfig,
    _TotalItersConfig,
)
from .enum import AnnealingStrategy, Mode, ThresholdMode

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

    @field_validator("milestones", mode="after")
    @classmethod
    def validator_milestones(cls, v):
        import numpy as np

        assert len(np.unique(v)) == len(v), "Epoch(s) in 'milestones' should be unique."
        return sorted(v)


class PolynomialLRConfig(LRSchedulerConfig, _TotalItersConfig, _LastEpochConfig):
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

    @field_validator("min_lr", mode="after")
    @classmethod
    def min_lr_validator(cls, v):
        """Checks that 'ELSE' is always in 'min_lr' if it is a dict."""
        return cls.group_validator(v, field_name="min_lr")


class OneCycleLRConfig(LRSchedulerConfig, _LastEpochConfig):
    """
    Config class for :py:class:`torch.optim.lr_scheduler.OneCycleLR`.
    """

    max_lr: Union[PositiveFloat, Dict[str, PositiveFloat]]
    total_steps: Optional[PositiveInt]
    epochs: Optional[PositiveInt]
    steps_per_epoch: Optional[PositiveInt]
    pct_start: NonNegativeFloat
    anneal_strategy: AnnealingStrategy
    cycle_momentum: bool
    base_momentum: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]]
    max_momentum: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]]
    div_factor: PositiveFloat
    final_div_factor: PositiveFloat
    three_phase: bool

    def __init__(
        self,
        max_lr: Union[PositiveFloat, Dict[str, PositiveFloat]],
        total_steps: Union[Optional[PositiveInt], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        epochs: Union[Optional[PositiveInt], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        steps_per_epoch: Union[Optional[PositiveInt], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        pct_start: Union[NonNegativeFloat, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        anneal_strategy: Union[AnnealingStrategy, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        cycle_momentum: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        base_momentum: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        max_momentum: Union[
            NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        div_factor: Union[PositiveFloat, DefaultFromLibrary] = (DefaultFromLibrary.YES),
        final_div_factor: Union[PositiveFloat, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        three_phase: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        last_epoch: Union[int, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            max_lr=max_lr,
            total_steps=total_steps,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )

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
