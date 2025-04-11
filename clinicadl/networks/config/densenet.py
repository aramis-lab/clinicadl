from typing import Any, Callable, Optional, Sequence, Union

import torch.nn as nn
from pydantic import PositiveFloat, PositiveInt

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import DefaultFromLibrary

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

    n_dense_layers: Sequence[PositiveInt]
    init_features: PositiveInt
    growth_rate: PositiveInt
    bottleneck_factor: PositiveInt

    def __init__(
        self,
        spatial_dims: PositiveInt,
        in_channels: PositiveInt,
        num_outputs: Optional[PositiveInt],
        n_dense_layers: Union[Sequence[PositiveInt], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        init_features: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        growth_rate: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
        bottleneck_factor: Union[
            PositiveInt, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_outputs=num_outputs,
            n_dense_layers=n_dense_layers,
            init_features=init_features,
            growth_rate=growth_rate,
            bottleneck_factor=bottleneck_factor,
            act=act,
            output_act=output_act,
            dropout=dropout,
        )


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
