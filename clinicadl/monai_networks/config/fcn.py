from __future__ import annotations

from typing import Optional, Tuple, Union

from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import NetworkConfig
from .utils.enum import ImplementedNetworks

__all__ = ["FullyConnectedNetConfig", "VarFullyConnectedNetConfig"]


class FullyConnectedNetConfig(NetworkConfig):
    """Config class for fully connected networks."""

    in_channels: PositiveInt
    out_channels: PositiveInt
    hidden_channels: Tuple[PositiveInt, ...]

    dropout: Union[
        Optional[NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.FCN

    @computed_field
    @property
    def dim(self) -> Optional[int]:
        """Dimension of the images."""
        return None

    @field_validator("dropout")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        return cls.base_validator_dropout(v)


class VarFullyConnectedNetConfig(NetworkConfig):
    """Config class for fully connected networks."""

    in_channels: PositiveInt
    out_channels: PositiveInt
    latent_size: PositiveInt
    encode_channels: Tuple[PositiveInt, ...]
    decode_channels: Tuple[PositiveInt, ...]

    dropout: Union[
        Optional[NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.VAR_FCN

    @computed_field
    @property
    def dim(self) -> Optional[int]:
        """Dimension of the images."""
        return None

    @field_validator("dropout")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        return cls.base_validator_dropout(v)
