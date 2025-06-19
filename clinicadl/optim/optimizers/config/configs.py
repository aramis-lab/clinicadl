from typing import Dict, List, Optional, Tuple, Union

import torch
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
)

from clinicadl.utils.factories import get_defaults_from

from .base import OptimizerConfig

__all__ = [
    "AdadeltaConfig",
    "AdagradConfig",
    "AdamConfig",
    "RMSpropConfig",
    "SGDConfig",
]
ADA_DELTA_DEFAULTS = get_defaults_from(torch.optim.Adadelta)
ADAGRAD_DEFAULTS = get_defaults_from(torch.optim.Adagrad)
ADAM_DEFAULTS = get_defaults_from(torch.optim.Adam)
RMSPROP_DEFAULTS = get_defaults_from(torch.optim.RMSprop)
SGD_DEFAULTS = get_defaults_from(torch.optim.SGD)


class AdadeltaConfig(OptimizerConfig):
    """
    Config class for :py:class:`torch.optim.Adadelta`.
    """

    lr: Union[PositiveFloat, Dict[str, PositiveFloat]] = ADA_DELTA_DEFAULTS["lr"]
    freeze: Optional[Union[str, List[str]]] = None
    rho: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = ADA_DELTA_DEFAULTS[
        "rho"
    ]
    eps: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = ADA_DELTA_DEFAULTS[
        "eps"
    ]
    weight_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat]
    ] = ADA_DELTA_DEFAULTS["weight_decay"]
    foreach: Union[Optional[bool], Dict[str, Optional[bool]]] = ADA_DELTA_DEFAULTS[
        "foreach"
    ]
    capturable: Union[bool, Dict[str, bool]] = ADA_DELTA_DEFAULTS["capturable"]
    maximize: Union[bool, Dict[str, bool]] = ADA_DELTA_DEFAULTS["maximize"]
    differentiable: Union[bool, Dict[str, bool]] = ADA_DELTA_DEFAULTS["differentiable"]

    @field_validator("rho")
    @classmethod
    def validator_rho(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class AdagradConfig(OptimizerConfig):
    """
    Config class for :py:class:`torch.optim.Adagrad`.
    """

    lr: Union[PositiveFloat, Dict[str, PositiveFloat]] = ADAGRAD_DEFAULTS["lr"]
    freeze: Optional[Union[str, List[str]]] = None
    lr_decay: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = ADAGRAD_DEFAULTS[
        "lr_decay"
    ]
    weight_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat]
    ] = ADAGRAD_DEFAULTS["weight_decay"]
    initial_accumulator_value: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat]
    ] = ADAGRAD_DEFAULTS["initial_accumulator_value"]
    eps: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = ADAGRAD_DEFAULTS["eps"]
    foreach: Union[Optional[bool], Dict[str, Optional[bool]]] = ADAGRAD_DEFAULTS[
        "foreach"
    ]
    maximize: Union[bool, Dict[str, bool]] = ADAGRAD_DEFAULTS["maximize"]
    differentiable: Union[bool, Dict[str, bool]] = ADAGRAD_DEFAULTS["differentiable"]
    fused: Union[Optional[bool], Dict[str, Optional[bool]]] = ADAGRAD_DEFAULTS["fused"]


class AdamConfig(OptimizerConfig):
    """
    Config class for :py:class:`torch.optim.Adam`.
    """

    lr: Union[PositiveFloat, Dict[str, PositiveFloat]] = ADAM_DEFAULTS["lr"]
    freeze: Optional[Union[str, List[str]]] = None
    betas: Union[
        Tuple[NonNegativeFloat, NonNegativeFloat],
        Dict[str, Tuple[NonNegativeFloat, NonNegativeFloat]],
    ] = ADAM_DEFAULTS["betas"]
    eps: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = ADAM_DEFAULTS["eps"]
    weight_decay: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = ADAM_DEFAULTS[
        "weight_decay"
    ]
    amsgrad: Union[bool, Dict[str, bool]] = ADAM_DEFAULTS["amsgrad"]
    foreach: Union[Optional[bool], Dict[str, Optional[bool]]] = ADAM_DEFAULTS["foreach"]
    maximize: Union[bool, Dict[str, bool]] = ADAM_DEFAULTS["maximize"]
    capturable: Union[bool, Dict[str, bool]] = ADAM_DEFAULTS["capturable"]
    differentiable: Union[bool, Dict[str, bool]] = ADAM_DEFAULTS["differentiable"]
    fused: Union[Optional[bool], Dict[str, Optional[bool]]] = ADAM_DEFAULTS["fused"]

    @field_validator("betas")
    @classmethod
    def validator_betas(cls, v, ctx):
        return cls.validator_proba(v, ctx)


class RMSpropConfig(OptimizerConfig):
    """
    Config class for :py:class:`torch.optim.RMSprop`.
    """

    lr: Union[PositiveFloat, Dict[str, PositiveFloat]] = RMSPROP_DEFAULTS["lr"]
    freeze: Optional[Union[str, List[str]]] = None
    alpha: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = RMSPROP_DEFAULTS[
        "alpha"
    ]
    eps: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = RMSPROP_DEFAULTS["eps"]
    weight_decay: Union[
        NonNegativeFloat, Dict[str, NonNegativeFloat]
    ] = RMSPROP_DEFAULTS["weight_decay"]
    momentum: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = RMSPROP_DEFAULTS[
        "momentum"
    ]
    centered: Union[bool, Dict[str, bool]] = RMSPROP_DEFAULTS["centered"]
    capturable: Union[bool, Dict[str, bool]] = RMSPROP_DEFAULTS["capturable"]
    foreach: Union[Optional[bool], Dict[str, Optional[bool]]] = RMSPROP_DEFAULTS[
        "foreach"
    ]
    maximize: Union[bool, Dict[str, bool]] = RMSPROP_DEFAULTS["maximize"]
    differentiable: Union[bool, Dict[str, bool]] = RMSPROP_DEFAULTS["differentiable"]


class SGDConfig(OptimizerConfig):
    """
    Config class for :py:class:`torch.optim.SGD`.
    """

    lr: Union[PositiveFloat, Dict[str, PositiveFloat]] = SGD_DEFAULTS["lr"]
    freeze: Optional[Union[str, List[str]]] = None
    momentum: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = SGD_DEFAULTS[
        "momentum"
    ]
    dampening: Union[float, Dict[str, float]] = SGD_DEFAULTS["dampening"]
    weight_decay: Union[NonNegativeFloat, Dict[str, NonNegativeFloat]] = SGD_DEFAULTS[
        "weight_decay"
    ]
    nesterov: Union[bool, Dict[str, bool]] = SGD_DEFAULTS["nesterov"]
    maximize: Union[bool, Dict[str, bool]] = SGD_DEFAULTS["maximize"]
    foreach: Union[Optional[bool], Dict[str, Optional[bool]]] = SGD_DEFAULTS["foreach"]
    differentiable: Union[bool, Dict[str, bool]] = SGD_DEFAULTS["differentiable"]
    fused: Union[Optional[bool], Dict[str, Optional[bool]]] = SGD_DEFAULTS["fused"]
