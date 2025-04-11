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
from clinicadl.utils.config import update_kwargs_with_defaults
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    NetworkConfig,
    _DropOutConfig,
    _FullyConvConfig,
    _OutputActConfig,
)

__all__ = [
    "MLPConfig",
    "ConvEncoderConfig",
    "ConvDecoderConfig",
]


class _BaseMLPConvConfig(_OutputActConfig, _DropOutConfig):
    """
    Base config class for MLP, ConvEncoder and ConvDecoder options.
    """

    act: Optional[ActivationParameters]
    bias: bool
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
    norm: Optional[NormalizationParameters]

    def __init__(
        self,
        hidden_dims: Sequence[PositiveInt],
        act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        norm: Union[Optional[NormalizationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        kwargs = locals()
        del kwargs["self"]
        kwargs = update_kwargs_with_defaults(kwargs, function=nets.MLP.__init__)
        super().__init__(**kwargs)

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

    def __init__(
        self,
        num_inputs: PositiveInt,
        num_outputs: PositiveInt,
        hidden_dims: Sequence[PositiveInt],
        act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        norm: Union[Optional[NormalizationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            hidden_dims=hidden_dims,
            act=act,
            output_act=output_act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )


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

    pooling: Optional[PoolingParameters]
    pooling_indices: Optional[Sequence[int]]

    def __init__(
        self,
        channels: Sequence[PositiveInt],
        kernel_size: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        stride: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        padding: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        dilation: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        pooling: Union[Optional[PoolingParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        pooling_indices: Union[Optional[Sequence[int]], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        norm: Union[Optional[ConvNormalizationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        kwargs = locals()
        del kwargs["self"]
        kwargs = update_kwargs_with_defaults(kwargs, function=nets.ConvEncoder.__init__)
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def check_pooling(self):
        checked_indices = self.check_pool_indices(self.pooling_indices)
        check_pool_layers(self.pooling, pooling_indices=checked_indices)

        return self


class ConvEncoderConfig(NetworkConfig, ConvEncoderOptions, _FullyConvConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ConvEncoder`.
    """

    def __init__(
        self,
        spatial_dims: PositiveInt,
        in_channels: PositiveInt,
        channels: Sequence[PositiveInt],
        kernel_size: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        stride: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        padding: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        dilation: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        pooling: Union[Optional[PoolingParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        pooling_indices: Union[Optional[Sequence[int]], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        norm: Union[Optional[ConvNormalizationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            pooling=pooling,
            pooling_indices=pooling_indices,
            act=act,
            output_act=output_act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

    @model_validator(mode="after")
    def check_dim(self):
        self.check_args_dim(self.spatial_dims)
        return self


class ConvDecoderOptions(_BaseConvOptions):
    """
    Config class for ConvDecoder when it is a submodule.
    See for example: :py:class:`clinicadl.networks.nn.Generator`
    """

    output_padding: ConvParameters
    unpooling: Optional[UnpoolingParameters]
    unpooling_indices: Optional[Sequence[int]]

    def __init__(
        self,
        channels: Sequence[PositiveInt],
        kernel_size: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        stride: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        padding: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        output_padding: Union[
            ConvParameters, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        dilation: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        unpooling: Union[Optional[UnpoolingParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        unpooling_indices: Union[Optional[Sequence[int]], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        norm: Union[Optional[ConvNormalizationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        kwargs = locals()
        del kwargs["self"]
        kwargs = update_kwargs_with_defaults(kwargs, function=nets.ConvDecoder.__init__)
        super().__init__(**kwargs)

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


class ConvDecoderConfig(NetworkConfig, ConvDecoderOptions, _FullyConvConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ConvDecoder`.
    """

    def __init__(
        self,
        spatial_dims: PositiveInt,
        in_channels: PositiveInt,
        channels: Sequence[PositiveInt],
        kernel_size: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        stride: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        padding: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        output_padding: Union[
            ConvParameters, DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        dilation: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES,
        unpooling: Union[Optional[UnpoolingParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        unpooling_indices: Union[Optional[Sequence[int]], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        norm: Union[Optional[ConvNormalizationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        dropout: Union[
            Optional[PositiveFloat], DefaultFromLibrary
        ] = DefaultFromLibrary.YES,
        bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES,
        adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super(ConvDecoderConfig, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            unpooling=unpooling,
            unpooling_indices=unpooling_indices,
            act=act,
            output_act=output_act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

    @model_validator(mode="after")
    def check_dim(self):
        self.check_args_dim(self.spatial_dims)
        return self
