from __future__ import annotations

from typing import Union

from pydantic import (
    PositiveInt,
    computed_field,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import VaryingDepthNetworkConfig
from .utils.enum import ImplementedNetworks

__all__ = ["UNetConfig", "AttentionUnetConfig"]


class UNetConfig(VaryingDepthNetworkConfig):
    """Config class for UNet."""

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    out_channels: PositiveInt
    adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.UNET

    @computed_field
    @property
    def dim(self) -> int:
        """Dimension of the images."""
        return self.spatial_dims

    @model_validator(mode="after")
    def channels_strides_validator(self):
        """Checks coherence between parameters."""
        n_layers = len(self.channels)
        assert (
            n_layers >= 2
        ), f"Channels must be at least of length 2. You passed {self.channels}."
        assert (
            len(self.strides) == n_layers - 1
        ), f"Length of strides must be equal to len(channels)-1. You passed channels={self.channels} and strides={self.strides}."
        for s in self.strides:
            assert self._check_dimensions(
                s
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for strides. You passed {s}."

        return self


class AttentionUnetConfig(UNetConfig):
    """Config class for Attention UNet."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.ATT_UNET
