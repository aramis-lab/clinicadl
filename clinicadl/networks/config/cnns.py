from typing import Optional, Sequence

from pydantic import PositiveInt, field_validator, model_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.autoencoder import check_unpooling_mode
from clinicadl.networks.nn.layers.utils import ActivationParameters, UnpoolingMode
from clinicadl.utils.factories import get_defaults_from

from .base import NetworkConfig, _InShapeConfig
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


class CNNConfig(NetworkConfig, _InShapeConfig):
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


class GeneratorConfig(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.Generator`.
    """

    latent_size: PositiveInt
    start_shape: Sequence[PositiveInt]
    conv_args: ConvDecoderOptions
    mlp_args: Optional[MLPOptions] = GENERATOR_DEFAULTS["mlp_args"]

    @field_validator("start_shape", mode="after")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that 'start_shape' corresponds to 1D, 2D or 3D images."""
        assert (
            2 <= len(v) <= 4
        ), f"'start_shape' must be of length 2 (1D), 3 (2D image) or 4 (3D image). Don't forget the channel dimension. Got: {v}."
        return v

    @model_validator(mode="after")
    def check_dim(self):
        _, *inter_size = self.start_shape
        spatial_dims = len(inter_size)
        self.conv_args.check_args_dim(spatial_dims)

        return self


class AutoEncoderConfig(NetworkConfig, _InShapeConfig):
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
