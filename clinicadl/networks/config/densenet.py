from typing import Optional, Sequence, Union

from pydantic import PositiveFloat, PositiveInt

from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    NetworkConfig,
    _DropOutConfig,
    _FullyConvConfig,
    _MandatoryActConfig,
    _OptionalNumOutputsConfig,
    _OutputActConfig,
    _PretrainedFromLiteratureConfig,
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
    _OptionalNumOutputsConfig,
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


class DenseNet121Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet121`.
    """


class DenseNet161Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet161`.
    """


class DenseNet169Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet169`.
    """


class DenseNet201Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet201`.
    """
