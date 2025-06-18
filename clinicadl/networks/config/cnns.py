from typing import Optional, Sequence, Union

from pydantic import PositiveInt, model_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.autoencoder import check_unpooling_mode
from clinicadl.networks.nn.layers.utils import ActivationParameters, UnpoolingMode
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import get_defaults_from

from .base import NetworkConfig, _InShapeConfig, _OutputActConfig
from .mlp_conv import (
    ConvDecoderOptions,
    ConvEncoderOptions,
    MLPOptions,
)

__all__ = ["CNNConfig", "GeneratorConfig", "AutoEncoderConfig", "VAEConfig"]

CNN_DEFAULTS = get_defaults_from(nets.CNN)
GENERATOR_DEFAULTS = get_defaults_from(nets.Generator)
AUTOENCODER_DEFAULTS = get_defaults_from(nets.AutoEncoder)
VAE_DEFAULTS = get_defaults_from(nets.VAE)


class _MLPArgsConfig(ClinicaDLConfig):
    """Config class for 'mlp_args' option."""

    mlp_args: Optional[MLPOptions]


class _LatentSizeConfig(ClinicaDLConfig):
    """Config class for 'latent_size' option."""

    latent_size: PositiveInt


class CNNConfig(NetworkConfig, _InShapeConfig, _MLPArgsConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.CNN`.
    """

    in_shape: Sequence[PositiveInt]
    num_outputs: PositiveInt
    conv_args: ConvEncoderOptions
    mlp_args: Optional[MLPOptions] = CNN_DEFAULTS["mlp_args"]

    @model_validator(mode="after")
    def check_dim(self):
        _, *input_size = self.in_shape
        spatial_dims = len(input_size)
        self.conv_args.check_args_dim(spatial_dims)

        return self


class GeneratorConfig(NetworkConfig, _LatentSizeConfig, _MLPArgsConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.Generator`.
    """

    latent_size: PositiveInt
    start_shape: Sequence[PositiveInt]
    conv_args: ConvDecoderOptions
    mlp_args: Optional[MLPOptions] = GENERATOR_DEFAULTS["mlp_args"]

    @model_validator(mode="after")
    def check_dim(self):
        _, *inter_size = self.start_shape
        spatial_dims = len(inter_size)
        self.conv_args.check_args_dim(spatial_dims)

        return self


class AutoEncoderConfig(
    NetworkConfig, _InShapeConfig, _LatentSizeConfig, _MLPArgsConfig, _OutputActConfig
):
    """
    Config class for :py:class:`clinicadl.networks.nn.AutoEncoder`.
    """

    in_shape: Sequence[PositiveInt]
    latent_size: PositiveInt
    conv_args: ConvEncoderOptions
    mlp_args: Optional[MLPOptions] = AUTOENCODER_DEFAULTS["mlp_args"]
    out_channels: Optional[PositiveInt] = AUTOENCODER_DEFAULTS["out_channels"]
    output_act: Optional[ActivationParameters] = AUTOENCODER_DEFAULTS["output_act"]
    unpooling_mode: UnpoolingMode = AUTOENCODER_DEFAULTS["unpooling_mode"]

    @model_validator(mode="after")
    def check_dim(self):
        _, *input_size = self.in_shape
        spatial_dims = len(input_size)
        self.conv_args.check_args_dim(spatial_dims)
        check_unpooling_mode(self.unpooling_mode, spatial_dims)

        return self


class VAEConfig(AutoEncoderConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.VAE`.
    """
