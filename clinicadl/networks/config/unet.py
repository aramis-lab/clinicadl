from typing import Sequence, Union

from pydantic import PositiveInt, computed_field, field_validator

from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    ImplementedNetwork,
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
    """Config class for UNet."""

    out_channels: PositiveInt
    channels: Union[Sequence[PositiveInt], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.UNET

    @field_validator("channels")
    @classmethod
    def channels_validator(cls, v):
        if v != DefaultFromLibrary.YES and isinstance(v, Sequence) and len(v) < 2:
            raise ValueError(f"length of channels must be no less than 2. Got {v}")
        return v


class AttentionUNetConfig(UNetConfig):
    """Config class for AttentionUNet."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.ATT_UNET
