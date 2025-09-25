from abc import abstractmethod
from collections.abc import Sequence

import torch.optim as optim
from pydantic import (
    field_validator,
    model_validator,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig

from .enum import LRSchedulerType
from .utils import is_dict_type


class LRSchedulerConfig(ObjectConfig):
    """Base config class for the LR scheduler."""

    @classmethod
    @abstractmethod
    def scheduler_type(cls) -> LRSchedulerType:
        """The type of LR scheduler (epoch-based, step-based, or metric-based)."""

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
        self._check_optimizer_consistency(optimizer)
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

    def _check_optimizer_consistency(self, optimizer: Optimizer) -> None:
        """
        Checks if LR scheduler and optimizers are consistent.
        """
        n_optimizer_groups = len(optimizer.param_groups)

        for field_name, field in type(self).model_fields.items():
            if is_dict_type(field.annotation):
                value = getattr(self, field_name)

                if isinstance(value, (Sequence, dict)):
                    n_groups = len(value)

                    if n_groups != n_optimizer_groups:
                        raise ValueError(
                            f"There are {n_optimizer_groups} parameter groups in the optimizer, "
                            f"but {n_groups} groups in the {self._get_name()} for parameter '{field_name}'. "
                            "Make sure that the parameter groups match between your optimizer and LR scheduler!"
                        )

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
