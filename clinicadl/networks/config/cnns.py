from typing import Optional, Sequence, Union

from pydantic import PositiveInt, computed_field, model_validator

from clinicadl.networks.nn.autoencoder import check_unpooling_mode
from clinicadl.networks.nn.layers.utils import UnpoolingMode
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetwork, NetworkConfig, _InShapeConfig, _OutputActConfig
from .mlp_conv import ConvDecoderOptions, ConvEncoderOptions, MLPOptions

__all__ = ["CNNConfig", "GeneratorConfig", "AutoEncoderConfig", "VAEConfig"]


class _MLPArgsConfig(NetworkConfig):
    """Base config class for networks with 'mlp_args' option."""

    mlp_args: Union[Optional[MLPOptions], DefaultFromLibrary] = DefaultFromLibrary.YES


class _LatentSizeConfig(NetworkConfig):
    """Base config class for networks with 'latent_size' option."""

    latent_size: PositiveInt


class CNNConfig(_InShapeConfig, _MLPArgsConfig):
    """Config class for CNN."""

    num_outputs: PositiveInt
    conv_args: ConvEncoderOptions

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.CNN

    @model_validator(mode="after")
    def check_dim(self):
        _, *input_size = self.in_shape
        spatial_dims = len(input_size)
        self.conv_args.check_args_dim(spatial_dims)

        return self


class GeneratorConfig(_LatentSizeConfig, _MLPArgsConfig):
    """Config class for Generator."""

    start_shape: Sequence[PositiveInt]
    conv_args: ConvDecoderOptions

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.GENERATOR

    @model_validator(mode="after")
    def check_dim(self):
        _, *inter_size = self.start_shape
        spatial_dims = len(inter_size)
        self.conv_args.check_args_dim(spatial_dims)

        return self


class AutoEncoderConfig(
    _InShapeConfig, _LatentSizeConfig, _MLPArgsConfig, _OutputActConfig
):
    """Config class for AutoEncoder."""

    conv_args: ConvEncoderOptions
    out_channels: Union[
        Optional[PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    unpooling_mode: Union[UnpoolingMode, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.AE

    @model_validator(mode="after")
    def check_dim(self):
        _, *input_size = self.in_shape
        spatial_dims = len(input_size)
        self.conv_args.check_args_dim(spatial_dims)
        if self.unpooling_mode != DefaultFromLibrary.YES:
            check_unpooling_mode(self.unpooling_mode, spatial_dims)

        return self


class VAEConfig(AutoEncoderConfig):
    """Config class for Variational AutoEncoder."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.VAE
