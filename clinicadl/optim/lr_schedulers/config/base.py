from typing import Set

import torch.optim as optim
from pydantic import (
    field_validator,
    model_validator,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig


class LRSchedulerConfig(ObjectConfig):
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

    @model_validator(mode="after")
    def check_groups_consistency(self):
        """
        Checks that parameter groups are the same across fields.
        """
        ref_groups = None
        ref_field = None
        for name, value in self:
            if isinstance(value, dict):
                groups = set(value.keys())
                if not ref_field:
                    ref_field = name
                    ref_groups = groups
                else:
                    if groups != ref_groups:
                        raise ValueError(
                            f"You passed different parameter groups to '{name}' ({groups}) "
                            f"and '{ref_field}' ({ref_groups}). You must pass the same groups "
                            "(the groups you passed to your optimizer)."
                        )

        return self

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
        config_dict = self.model_dump(exclude={"name"})

        # deal with parameter groups
        for arg, value in config_dict.items():
            if isinstance(value, dict):
                list_values = [
                    value[group] for group in sorted(value.keys()) if group != "ELSE"
                ]  # order in the list is important
                list_values.append(value["ELSE"])  # ELSE must be the last group
                config_dict[arg] = list_values

        return associated_class(optimizer, **config_dict)

    def get_all_groups(self) -> Set[str]:
        """
        Returns all parameter groups mentioned by the user in the fields.

        Returns
        -------
        Set[str]
            The groups.
        """
        for _, value in self:
            if isinstance(value, dict):
                return set(value.keys())  # all dict have the same keys

        return set()

    @classmethod
    def _get_class(cls) -> type[optim.lr_scheduler.LRScheduler]:
        """Returns the lr scheduler associated to this config class."""
        return getattr(optim.lr_scheduler, cls._get_name())


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
