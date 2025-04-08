from typing import Set

from pydantic import (
    PositiveFloat,
    PositiveInt,
    field_validator,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clinicadl.utils.config import ClinicaDLConfig, NewClinicaDLConfig


class LRSchedulerConfig(NewClinicaDLConfig):
    """Base config class for the LR scheduler."""

    @classmethod
    def group_validator(cls, v, field_name: str):
        """Checks that 'ELSE' is always in a field if it is a dict (i.e. if parameter groups are passed)."""
        if isinstance(v, dict) and "ELSE" not in v:
            raise ValueError(
                f"If you pass a dict to '{field_name}', it must contain the key 'ELSE', that corresponds "
                f"to the value applied to the rest of the parameters. Got: {v}"
            )
        return v

    def get_object(self, optimizer: Optimizer) -> LRScheduler:  # pylint: disable=arguments-differ
        """
        Returns the LR scheduler associated to this configuration,
        parametrized with the parameters passed by the user.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to schedule.

        Returns
        -------
        torch.optim.lr_scheduler.LRScheduler
            The PyTorch LR Scheduler, associated to the optimizer.
        """
        associated_class = self._get_class()
        return associated_class(optimizer, **self.model_dump(exclude=""))

    def get_all_groups(self) -> Set[str]:
        """
        Returns all parameter groups mentioned by the user in the fields.

        Returns
        -------
        Set[str]
            The groups.
        """
        groups = set()
        for _, value in self:
            if isinstance(value, dict):
                groups.update(set(value.keys()))

        return groups


class _GammaConfig(ClinicaDLConfig):
    """Config class for 'gamma' parameter."""

    gamma: PositiveFloat


class _FactorConfig(ClinicaDLConfig):
    """Config class for 'factor' parameter."""

    factor: PositiveFloat


class _TotalItersConfig(ClinicaDLConfig):
    """Config class for 'total_iters' parameter."""

    total_iters: PositiveInt


class _LastEpochConfig(ClinicaDLConfig):
    """Config class for 'last_epoch' parameter."""

    last_epoch: int

    @field_validator("last_epoch")
    @classmethod
    def validator_last_epoch(cls, v):
        if isinstance(v, int):
            assert (
                -1 <= v
            ), f"last_epoch must be -1 or a non-negative int but it has been set to {v}."
        return v
