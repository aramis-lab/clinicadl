from typing import Optional, Sequence, Union

from pydantic import PositiveInt, computed_field

from clinicadl.networks.nn.layers.utils import (
    ActivationParameters,
    UnpoolingMode,
)
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetwork, NetworkConfig
from .conv_encoder import ConvEncoderOptions
from .mlp import MLPOptions


class AutoEncoderConfig(NetworkConfig):
    """Config class for AutoEncoder."""

    in_shape: Sequence[PositiveInt]
    latent_size: PositiveInt
    conv_args: ConvEncoderOptions
    mlp_args: Union[Optional[MLPOptions], DefaultFromLibrary] = DefaultFromLibrary.YES
    out_channels: Union[
        Optional[PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    unpooling_mode: Union[UnpoolingMode, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.AE


class VAEConfig(AutoEncoderConfig):
    """Config class for Variational AutoEncoder."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.VAE
