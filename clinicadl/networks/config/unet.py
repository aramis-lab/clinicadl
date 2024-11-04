from typing import Optional, Sequence, Union

from pydantic import PositiveFloat, PositiveInt, computed_field

from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetwork, NetworkConfig


class UNetConfig(NetworkConfig):
    """Config class for UNet."""

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    out_channels: PositiveInt
    channels: Union[Sequence[PositiveInt], DefaultFromLibrary] = DefaultFromLibrary.YES
    act: Union[ActivationParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    dropout: Union[Optional[PositiveFloat], DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.UNET


class AttentionUNetConfig(UNetConfig):
    """Config class for AttentionUNet."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.ATT_UNET
