from typing import Sequence, Union

from pydantic import PositiveInt, computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    ImplementedNetwork,
    NetworkType,
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
    _FullyConvConfig,
    _OptionalLastLinearLayersConfig,
    _MandatoryActConfig,
    _OutputActConfig,
    _DropOutConfig,
):
    """Config class for DenseNet."""

    n_dense_layers: Union[
        Sequence[PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    init_features: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    growth_rate: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    bottleneck_factor: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.DENSENET


class _PreTrainedDenseNetConfig(_PreTrainedConfig):
    """Base config class for SOTA DenseNets."""

    @computed_field
    @property
    def _type(self) -> NetworkType:
        """To know where to look for the network."""
        return NetworkType.DENSENET


class DenseNet121Config(_PreTrainedDenseNetConfig):
    """Config class for DenseNet-121."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.DENSENET_121


class DenseNet161Config(_PreTrainedDenseNetConfig):
    """Config class for DenseNet-161."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.DENSENET_161


class DenseNet169Config(_PreTrainedDenseNetConfig):
    """Config class for DenseNet-169."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.DENSENET_169


class DenseNet201Config(_PreTrainedDenseNetConfig):
    """Config class for DenseNet-201."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.DENSENET_201
