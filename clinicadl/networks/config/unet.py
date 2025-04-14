from typing import Optional, Sequence, Union

from pydantic import PositiveFloat, PositiveInt, field_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.config import update_kwargs_with_defaults
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    NetworkConfig,
    _DropOutConfig,
    _FullyConvConfig,
    _MandatoryActConfig,
    _OutputActConfig,
)

__all__ = [
    "UNetConfig",
    "AttentionUNetConfig",
]


class UNetConfig(
    NetworkConfig,
    _FullyConvConfig,
    _MandatoryActConfig,
    _OutputActConfig,
    _DropOutConfig,
):
    """
    Config class for :py:class:`clinicadl.networks.nn.UNet`.
    """

    out_channels: PositiveInt
    channels: Union[Sequence[PositiveInt], DefaultFromLibrary] = DefaultFromLibrary.YES

    def __init__(
        self,
        spatial_dims: PositiveInt,
        in_channels: PositiveInt,
        out_channels: PositiveInt,
        channels: Union[
            Sequence[PositiveInt], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        act: Union[ActivationParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
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
            out_channels=out_channels,
            channels=channels,
            act=act,
            output_act=output_act,
            dropout=dropout,
        )

    @field_validator("channels")
    @classmethod
    def channels_validator(cls, v):
        if isinstance(v, Sequence) and len(v) < 2:
            raise ValueError(f"length of channels must be no less than 2. Got {v}")
        return v


class AttentionUNetConfig(UNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.AttentionUNet`.
    """

    def __init__(
        self,
        spatial_dims: PositiveInt,
        in_channels: PositiveInt,
        out_channels: PositiveInt,
        channels: Union[
            Sequence[PositiveInt], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        act: Union[ActivationParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        kwargs = locals()
        del kwargs["self"]
        kwargs = update_kwargs_with_defaults(kwargs, function=nets.UNet.__init__)
        super(UNetConfig, self).__init__(**kwargs)
