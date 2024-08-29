from typing import Optional, Tuple, Union

from pydantic import (
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import VaryingDepthNetworkConfig
from .utils.enum import ImplementedNetworks

__all__ = ["AutoEncoderConfig", "VarAutoEncoderConfig"]


class AutoEncoderConfig(VaryingDepthNetworkConfig):
    """Config class for autoencoders."""

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    out_channels: PositiveInt

    inter_channels: Union[
        Optional[Tuple[PositiveInt, ...]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    inter_dilations: Union[
        Optional[Tuple[PositiveInt, ...]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    num_inter_units: Union[NonNegativeInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    padding: Union[
        Optional[Union[PositiveInt, Tuple[PositiveInt, ...]]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.AE

    @computed_field
    @property
    def dim(self) -> int:
        """Dimension of the images."""
        return self.spatial_dims

    @model_validator(mode="after")
    def model_validator(self):
        """Checks coherence between parameters."""
        if self.padding != DefaultFromLibrary.YES:
            assert self._check_dimensions(
                self.padding
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for padding. You passed {self.padding}."
        if isinstance(self.inter_channels, tuple) and isinstance(
            self.inter_dilations, tuple
        ):
            assert len(self.inter_channels) == len(
                self.inter_dilations
            ), "inter_channels and inter_dilations muust have the same size."
        elif isinstance(self.inter_dilations, tuple) and not isinstance(
            self.inter_channels, tuple
        ):
            raise ValueError(
                "You passed inter_dilations but didn't pass inter_channels."
            )
        return self


class VarAutoEncoderConfig(AutoEncoderConfig):
    """Config class for variational autoencoders."""

    in_shape: Tuple[PositiveInt, ...]
    in_channels: Optional[int] = None
    latent_size: PositiveInt
    use_sigmoid: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.VAE

    @model_validator(mode="after")
    def model_validator_bis(self):
        """Checks coherence between parameters."""
        assert (
            len(self.in_shape[1:]) == self.spatial_dims
        ), f"You passed {self.spatial_dims} for spatial_dims, but in_shape suggests {len(self.in_shape[1:])} spatial dimensions."
