from typing import Optional, Sequence

from pydantic import PositiveFloat, PositiveInt

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import get_defaults_from

from .base import (
    NetworkConfig,
    _DropoutConfig,
    _SpatialDimsConfig,
)

__all__ = [
    "DenseNetConfig",
    "DenseNet121Config",
    "DenseNet161Config",
    "DenseNet169Config",
    "DenseNet201Config",
]

DENSE_NET_DEFAULTS = get_defaults_from(nets.DenseNet)
DENSE_NET_121_DEFAULTS = get_defaults_from(nets.DenseNet121)
DENSE_NET_161_DEFAULTS = get_defaults_from(nets.DenseNet161)
DENSE_NET_169_DEFAULTS = get_defaults_from(nets.DenseNet169)
DENSE_NET_201_DEFAULTS = get_defaults_from(nets.DenseNet201)


class DenseNetConfig(
    NetworkConfig,
    _SpatialDimsConfig,
    _DropoutConfig,
):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    num_outputs: Optional[PositiveInt]
    n_dense_layers: Sequence[PositiveInt] = DENSE_NET_DEFAULTS["n_dense_layers"]
    init_features: PositiveInt = DENSE_NET_DEFAULTS["init_features"]
    growth_rate: PositiveInt = DENSE_NET_DEFAULTS["growth_rate"]
    bottleneck_factor: PositiveInt = DENSE_NET_DEFAULTS["bottleneck_factor"]
    act: Optional[ActivationParameters] = DENSE_NET_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = DENSE_NET_DEFAULTS["output_act"]
    dropout: Optional[PositiveFloat] = DENSE_NET_DEFAULTS["dropout"]


class DenseNet121Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet121`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = DENSE_NET_121_DEFAULTS["output_act"]
    pretrained: bool = DENSE_NET_121_DEFAULTS["pretrained"]


class DenseNet161Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet161`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = DENSE_NET_161_DEFAULTS["output_act"]
    pretrained: bool = DENSE_NET_161_DEFAULTS["pretrained"]


class DenseNet169Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet169`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = DENSE_NET_169_DEFAULTS["output_act"]
    pretrained: bool = DENSE_NET_169_DEFAULTS["pretrained"]


class DenseNet201Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet201`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = DENSE_NET_201_DEFAULTS["output_act"]
    pretrained: bool = DENSE_NET_201_DEFAULTS["pretrained"]
