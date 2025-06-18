from typing import Any, Callable, Optional, Sequence

import torch.nn as nn
from pydantic import PositiveFloat, PositiveInt

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import get_defaults_from

from .base import (
    ImplementedNetwork,
    NetworkConfig,
    _DropOutConfig,
    _FullyConvConfig,
    _MandatoryActConfig,
    _OptionalLastLinearLayersConfig,
    _OutputActConfig,
    _PreTrainedConfig,
)

DENSENET_DEFAULTS = get_defaults_from(nets.DenseNet)

__all__ = [
    "DenseNetConfig",
    "DenseNet121Config",
    "DenseNet161Config",
    "DenseNet169Config",
    "DenseNet201Config",
]


class DenseNetConfig(
    NetworkConfig,
    _FullyConvConfig,
    _OptionalLastLinearLayersConfig,
    _MandatoryActConfig,
    _OutputActConfig,
    _DropOutConfig,
):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    num_outputs: Optional[PositiveInt]
    n_dense_layers: Sequence[PositiveInt] = DENSENET_DEFAULTS["n_dense_layers"]
    init_features: PositiveInt = DENSENET_DEFAULTS["init_features"]
    growth_rate: PositiveInt = DENSENET_DEFAULTS["growth_rate"]
    bottleneck_factor: PositiveInt = DENSENET_DEFAULTS["bottleneck_factor"]
    act: Optional[ActivationParameters] = DENSENET_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = DENSENET_DEFAULTS["output_act"]
    dropout: Optional[PositiveFloat] = DENSENET_DEFAULTS["dropout"]


class _PreTrainedDenseNetConfig(_PreTrainedConfig):
    """Base config class for SOTA DenseNets."""

    @classmethod
    def _get_class(cls) -> Callable[[Any], nn.Module]:
        """Returns the network associated to this config class."""
        return nets.get_densenet


class DenseNet121Config(_PreTrainedDenseNetConfig):
    """
    Config class for :py:func:`DenseNet-121 <clinicadl.networks.nn.get_densenet>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.DENSENET_121.value


class DenseNet161Config(_PreTrainedDenseNetConfig):
    """
    Config class for :py:func:`DenseNet-161 <clinicadl.networks.nn.get_densenet>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.DENSENET_161.value


class DenseNet169Config(_PreTrainedDenseNetConfig):
    """
    Config class for :py:func:`DenseNet-169 <clinicadl.networks.nn.get_densenet>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.DENSENET_169.value


class DenseNet201Config(_PreTrainedDenseNetConfig):
    """
    Config class for :py:func:`DenseNet-201 <clinicadl.networks.nn.get_densenet>`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.DENSENET_201.value
