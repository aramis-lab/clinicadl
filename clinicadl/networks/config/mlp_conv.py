from typing import Optional, Sequence, Union

from pydantic import (
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.conv_decoder import check_unpool_layers
from clinicadl.networks.nn.conv_encoder import check_pool_layers
from clinicadl.networks.nn.layers.utils import (
    ActivationParameters,
    ConvNormalizationParameters,
    ConvParameters,
    NormalizationParameters,
    PoolingParameters,
    UnpoolingParameters,
)
from clinicadl.networks.nn.utils import (
    check_adn_ordering,
    check_norm_layer,
    check_pool_indices,
    ensure_list_of_tuples,
)
from clinicadl.utils.factories import get_defaults_from

from .base import NetworkConfig, _DropoutConfig, _SpatialDimsConfig

MLP_DEFAULTS = get_defaults_from(nets.MLP)
CONV_ENCODER_DEFAULTS = get_defaults_from(nets.ConvEncoder)
CONV_DECODER_DEFAULTS = get_defaults_from(nets.ConvDecoder)

__all__ = [
    "MLPConfig",
    "ConvEncoderConfig",
    "ConvDecoderConfig",
]


class _BaseMLPConvConfig(_DropoutConfig):
    """
    Base config class for MLP, ConvEncoder and ConvDecoder options.
    """

    norm: Optional[Union[NormalizationParameters, ConvNormalizationParameters]]
    adn_ordering: str

    @field_validator("adn_ordering")
    @classmethod
    def adn_ordering_validator(cls, v):
        return check_adn_ordering(v)

    @classmethod
    def base_norm_validator(cls, v):
        return check_norm_layer(v)


class MLPOptions(_BaseMLPConvConfig):
    """
    Config class for MLP when it is a submodule.
    See for example: :py:class:`clinicadl.networks.nn.CNN`
    """

    hidden_dims: Sequence[PositiveInt]
    act: Optional[ActivationParameters] = MLP_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = MLP_DEFAULTS["output_act"]
    norm: Optional[NormalizationParameters] = MLP_DEFAULTS["norm"]
    dropout: Optional[PositiveFloat] = MLP_DEFAULTS["dropout"]
    bias: bool = MLP_DEFAULTS["bias"]
    adn_ordering: str = MLP_DEFAULTS["adn_ordering"]

    @field_validator("norm")
    @classmethod
    def norm_validator(cls, v):
        return cls.base_norm_validator(v)


class MLPConfig(NetworkConfig, MLPOptions):
    """
    Config class for :py:class:`clinicadl.networks.nn.MLP`.
    """

    num_inputs: PositiveInt
    num_outputs: PositiveInt


class _BaseConvOptions(_BaseMLPConvConfig):
    """
    Base config class for ConvEncoder and ConvDecoder options.
    """

    channels: Sequence[PositiveInt]
    kernel_size: ConvParameters
    stride: ConvParameters
    padding: ConvParameters
    dilation: ConvParameters
    norm: Optional[ConvNormalizationParameters]

    @field_validator("norm")
    @classmethod
    def norm_validator(cls, v):
        return cls.base_norm_validator(v)

    def check_args_dim(self, dim: int) -> None:
        n_layers = len(self.channels)
        ensure_list_of_tuples(self.kernel_size, dim, n_layers, "kernel_size")
        ensure_list_of_tuples(self.stride, dim, n_layers, "stride")
        ensure_list_of_tuples(self.padding, dim, n_layers, "padding")
        ensure_list_of_tuples(self.dilation, dim, n_layers, "dilation")

    def check_pool_indices(self, indices: Optional[Sequence[int]]) -> Sequence[int]:
        return check_pool_indices(indices, n_layers=len(self.channels))


class ConvEncoderOptions(_BaseConvOptions):
    """
    Config class for ConvEncoder when it is a submodule.
    See for example: :py:class:`clinicadl.networks.nn.CNN`
    """

    channels: Sequence[PositiveInt]
    kernel_size: ConvParameters = CONV_ENCODER_DEFAULTS["kernel_size"]
    stride: ConvParameters = CONV_ENCODER_DEFAULTS["stride"]
    padding: ConvParameters = CONV_ENCODER_DEFAULTS["padding"]
    dilation: ConvParameters = CONV_ENCODER_DEFAULTS["dilation"]
    pooling: Optional[PoolingParameters] = CONV_ENCODER_DEFAULTS["pooling"]
    pooling_indices: Optional[Sequence[int]] = CONV_ENCODER_DEFAULTS["pooling_indices"]
    act: Optional[ActivationParameters] = CONV_ENCODER_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = CONV_ENCODER_DEFAULTS["output_act"]
    norm: Optional[ConvNormalizationParameters] = CONV_ENCODER_DEFAULTS["norm"]
    dropout: Optional[PositiveFloat] = CONV_ENCODER_DEFAULTS["dropout"]
    bias: bool = CONV_ENCODER_DEFAULTS["bias"]
    adn_ordering: str = CONV_ENCODER_DEFAULTS["adn_ordering"]

    @model_validator(mode="after")
    def check_pooling(self):
        checked_indices = self.check_pool_indices(self.pooling_indices)
        check_pool_layers(self.pooling, pooling_indices=checked_indices)

        return self


class ConvEncoderConfig(NetworkConfig, ConvEncoderOptions, _SpatialDimsConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ConvEncoder`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt

    @model_validator(mode="after")
    def check_dim(self):
        self.check_args_dim(self.spatial_dims)
        return self


class ConvDecoderOptions(_BaseConvOptions):
    """
    Config class for ConvDecoder when it is a submodule.
    See for example: :py:class:`clinicadl.networks.nn.Generator`
    """

    channels: Sequence[PositiveInt]
    kernel_size: ConvParameters = CONV_DECODER_DEFAULTS["kernel_size"]
    stride: ConvParameters = CONV_DECODER_DEFAULTS["stride"]
    padding: ConvParameters = CONV_DECODER_DEFAULTS["padding"]
    output_padding: ConvParameters = CONV_DECODER_DEFAULTS["output_padding"]
    dilation: ConvParameters = CONV_DECODER_DEFAULTS["dilation"]
    unpooling: Optional[UnpoolingParameters] = CONV_DECODER_DEFAULTS["unpooling"]
    unpooling_indices: Optional[Sequence[int]] = CONV_DECODER_DEFAULTS[
        "unpooling_indices"
    ]
    act: Optional[ActivationParameters] = CONV_DECODER_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = CONV_DECODER_DEFAULTS["output_act"]
    norm: Optional[ConvNormalizationParameters] = CONV_DECODER_DEFAULTS["norm"]
    dropout: Optional[PositiveFloat] = CONV_DECODER_DEFAULTS["dropout"]
    bias: bool = CONV_DECODER_DEFAULTS["bias"]
    adn_ordering: str = CONV_DECODER_DEFAULTS["adn_ordering"]

    @model_validator(mode="after")
    def check_unpooling(self):
        checked_indices = self.check_pool_indices(self.unpooling_indices)
        check_unpool_layers(self.unpooling, unpooling_indices=checked_indices)

        return self

    def check_args_dim(self, dim: int) -> None:
        super().check_args_dim(dim)
        ensure_list_of_tuples(
            self.output_padding, dim, len(self.channels), "output_padding"
        )


class ConvDecoderConfig(NetworkConfig, ConvDecoderOptions, _SpatialDimsConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ConvDecoder`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt

    @model_validator(mode="after")
    def check_dim(self):
        self.check_args_dim(self.spatial_dims)
        return self
