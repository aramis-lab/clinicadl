from typing import Optional, Sequence, Union

from pydantic import PositiveInt, model_validator

from clinicadl.networks.nn.autoencoder import check_unpooling_mode
from clinicadl.networks.nn.layers.utils import ActivationParameters, UnpoolingMode
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.factories import DefaultFromLibrary

from .base import NetworkConfig, _InShapeConfig, _OutputActConfig
from .mlp_conv import (
    ConvDecoderOptions,
    ConvEncoderOptions,
    MLPOptions,
)

__all__ = ["CNNConfig", "GeneratorConfig", "AutoEncoderConfig", "VAEConfig"]


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

    num_outputs: PositiveInt
    conv_args: ConvEncoderOptions

    def __init__(
        self,
        in_shape: Sequence[PositiveInt],
        num_outputs: PositiveInt,
        conv_args: ConvEncoderOptions,
        mlp_args: Union[
            Optional[MLPOptions], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            in_shape=in_shape,
            num_outputs=num_outputs,
            conv_args=conv_args,
            mlp_args=mlp_args,
        )

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

    start_shape: Sequence[PositiveInt]
    conv_args: ConvDecoderOptions

    def __init__(
        self,
        latent_size: PositiveInt,
        start_shape: Sequence[PositiveInt],
        conv_args: ConvDecoderOptions,
        mlp_args: Union[
            Optional[MLPOptions], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            latent_size=latent_size,
            start_shape=start_shape,
            conv_args=conv_args,
            mlp_args=mlp_args,
        )

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

    conv_args: ConvEncoderOptions
    out_channels: Optional[PositiveInt]
    unpooling_mode: UnpoolingMode

    def __init__(
        self,
        in_shape: Sequence[PositiveInt],
        latent_size: PositiveInt,
        conv_args: ConvEncoderOptions,
        mlp_args: Union[
            Optional[MLPOptions], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        out_channels: Union[Optional[PositiveInt], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        unpooling_mode: Union[
            UnpoolingMode, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            in_shape=in_shape,
            latent_size=latent_size,
            conv_args=conv_args,
            mlp_args=mlp_args,
            out_channels=out_channels,
            output_act=output_act,
            unpooling_mode=unpooling_mode,
        )

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
