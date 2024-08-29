from __future__ import annotations

from typing import Tuple, Union

from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import NetworkConfig
from .utils.enum import ImplementedNetworks

__all__ = ["DenseNetConfig"]


class DenseNetConfig(NetworkConfig):
    """Config class for DenseNet."""

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    out_channels: PositiveInt
    init_features: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    growth_rate: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    block_config: Union[
        Tuple[PositiveInt, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    bn_size: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    dropout_prob: Union[NonNegativeFloat, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.DENSE_NET

    @computed_field
    @property
    def dim(self) -> int:
        """Dimension of the images."""
        return self.spatial_dims

    @field_validator("dropout_prob")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        return cls.base_validator_dropout(v)
