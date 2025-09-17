from enum import Enum
from typing import List, Optional, Set, Union

import torch.nn as nn
import torch.optim as optim
from pydantic import (
    field_validator,
    model_validator,
)

from clinicadl.utils.config import ObjectConfig

from .utils import (
    get_params_in_groups,
    get_params_not_in_groups,
    regroup_args_by_param_group,
)

__all__ = [
    "ImplementedOptimizer",
    "OptimizerConfig",
]


class ImplementedOptimizer(str, Enum):
    """Implemented optimizers in ClinicaDL."""

    ADADELTA = "Adadelta"
    ADAGRAD = "Adagrad"
    ADAM = "Adam"
    RMS_PROP = "RMSprop"
    SGD = "SGD"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented optimizers are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class OptimizerConfig(ObjectConfig):
    """Base config class for the optimizer."""

    freeze: Optional[Union[str, List[str]]] = None

    def get_object(self, network: nn.Module) -> optim.Optimizer:  # pylint: disable=arguments-differ
        """
        Returns the optimizer associated to this configuration,
        parametrized with the parameters passed by the user.

        Parameters
        ----------
        network : torch.nn.Module
            The neural network to optimize.

        Returns
        -------
        torch.optim.Optimizer
            The PyTorch optimizer.
        """

        freeze = [] if self.freeze is None else self.freeze

        to_freeze, _ = get_params_in_groups(network, groups=freeze)
        for param in to_freeze:
            param.requires_grad = False

        config_dict = self.model_dump(exclude={"name", "freeze"})

        # deal with parameter groups
        args_by_group, args_global = regroup_args_by_param_group(config_dict)
        if len(args_by_group) == 0:  # no parameter groups
            params = network.parameters()
        else:
            params = []
            args_by_group = sorted(
                args_by_group.items()
            )  # order in the list is important to match lr_scheduler
            for group, args in args_by_group:
                params_in_group, _ = get_params_in_groups(network, group)
                args.update({"params": params_in_group})
                params.append(args)

            other_params, other_param_names = get_params_not_in_groups(
                network, groups=[group for group, _ in args_by_group]
            )
            if len(other_param_names) > 0:
                params.append({"params": other_params})

        associated_class = self._get_class()

        return associated_class(params, **args_global)

    @classmethod
    def _get_class(cls) -> type[optim.Optimizer]:
        """Returns the optimizer associated to this config class."""
        return getattr(optim, cls._get_name())

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
