from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary


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


class OptimizerConfig(BaseModel):
    """Config class to configure the optimizer."""

    optimizer: ImplementedOptimizer = ImplementedOptimizer.ADAM

    lr: Union[
        PositiveFloat, Dict[str, PositiveFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    weight_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    betas: Union[
        Tuple[NonNegativeFloat, NonNegativeFloat],
        Dict[str, Tuple[NonNegativeFloat, NonNegativeFloat]],
        DefaultFromLibrary,
    ] = DefaultFromLibrary.YES
    alpha: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    momentum: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    rho: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    lr_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    eps: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    dampening: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    initial_accumulator_value: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    centered: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES
    nesterov: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES
    amsgrad: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES
    foreach: Union[
        Optional[bool], Dict[str, Optional[bool]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    maximize: Union[bool, Dict[str, bool], DefaultFromLibrary] = DefaultFromLibrary.YES
    differentiable: Union[
        bool, Dict[str, bool], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    capturable: Union[
        bool, Dict[str, bool], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    fused: Union[
        Optional[bool], Dict[str, bool], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )

    @field_validator("betas", "rho", "alpha", "dampening")
    @classmethod
    def validator_proba(cls, v, ctx):
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

    def get_all_groups(self) -> List[str]:
        """
        Returns all groups mentioned by the user in the fields.

        Returns
        -------
        List[str]
            The list of groups.
        """
        groups = set()
        for _, value in self.model_dump().items():
            if isinstance(value, dict):
                groups.update(set(value.keys()))

        return list(groups)
