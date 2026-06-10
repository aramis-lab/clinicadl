from typing import Dict, List, Optional, Tuple, Union

import torch
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
)

from clinicadl.utils.doc import add_suffix_to_doc
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

DOCUMENT_EXTRA_PARAMETERS = """
The parameters of the optimizer can here be passed via dictionaries, whose keys are
parameter groups and values are the values to apply to these groups. Such a dictionary
must always contain the key ``"ELSE"`` that specifies the value for the rest of the parameters.

``freeze`` can be used to freeze some weights of the neural network.

Examples
--------
.. code-block::

    >>> from clinicadl.networks.nn import CNN
    >>> network = CNN(
            in_shape=(1, 16, 16, 16),
            num_outputs=1,
            conv_args={"channels": [2, 4]},
        )
    >>> network
    CNN(
        (convolutions): ConvEncoder(
            (layer0): Convolution(
                (conv): Conv3d(1, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                (adn): ADN(
                    (N): InstanceNorm3d(2, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
                    (A): PReLU(num_parameters=1)
                )
            )
            (layer1): Convolution(
                (conv): Conv3d(2, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
        )
        (mlp): MLP(
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (output): Sequential(
                (linear): Linear(in_features=6912, out_features=1, bias=True)
            )
        )
    )
    >>> from clinicadl.optim.optimizers.config import AdamConfig
    >>> optimizer_config = AdamConfig(
            freeze="mlp.output", lr={"convolutions.layer0": 1e-2, "ELSE": 1e-3}
        )
    >>> optimizer = optimizer_config.get_object(network)
    >>> len(optimizer.param_groups)
    2   # 2 groups of parameters: 'convolutions.layer0' and the rest of the network
    >>> next(net.mlp.output.parameters()).requires_grad
    False   # 'mlp.output' is frozen
"""


@add_suffix_to_doc(DOCUMENT_EXTRA_PARAMETERS)
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
        return cls._validator_proba(v, ctx)


@add_suffix_to_doc(DOCUMENT_EXTRA_PARAMETERS)
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


@add_suffix_to_doc(DOCUMENT_EXTRA_PARAMETERS)
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
        return cls._validator_proba(v, ctx)


@add_suffix_to_doc(DOCUMENT_EXTRA_PARAMETERS)
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


@add_suffix_to_doc(DOCUMENT_EXTRA_PARAMETERS)
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
