from typing import Optional, Sequence, Union

from pydantic import PositiveInt, computed_field, field_validator, model_validator

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
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    ImplementedNetwork,
    NetworkConfig,
    _DropOutConfig,
    _FullyConvConfig,
    _OutputActConfig,
)

__all__ = [
    "MLPOptions",
    "MLPConfig",
    "ConvEncoderOptions",
    "ConvDecoderOptions",
    "ConvEncoderConfig",
    "ConvDecoderConfig",
]


class _BaseMLPConvConfig(_OutputActConfig, _DropOutConfig):
    """
    Base config class for MLP, ConvEncoder and ConvDecoder options.
    """

    act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES

    @field_validator("adn_ordering")
    @classmethod
    def adn_ordering_validator(cls, v):
        if v != DefaultFromLibrary.YES:
            return check_adn_ordering(v)
        return v

    @classmethod
    def base_norm_validator(cls, v):
        if v != DefaultFromLibrary.YES:
            return check_norm_layer(v)
        return v


class MLPOptions(_BaseMLPConvConfig):
    """
    Config class for MLP when it is a submodule.
    See for example: :py:class:`clinicadl.monai_networks.nn.cnn.CNN`
    """

    hidden_dims: Sequence[PositiveInt]
    norm: Union[
        Optional[NormalizationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @field_validator("norm")
    @classmethod
    def norm_validator(cls, v):
        return cls.base_norm_validator(v)


class MLPConfig(NetworkConfig, MLPOptions):
    """Config class for Multi Layer Perceptron."""

    num_inputs: PositiveInt
    num_outputs: PositiveInt

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.MLP


class _ConvOptions(_BaseMLPConvConfig):
    """
    Base config class for ConvEncoder and ConvDecoder options.
    """

    channels: Sequence[PositiveInt]
    kernel_size: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    stride: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    padding: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    dilation: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    norm: Union[
        Optional[ConvNormalizationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @field_validator("norm")
    @classmethod
    def norm_validator(cls, v):
        return cls.base_norm_validator(v)

    def check_args_dim(self, dim: int) -> None:
        n_layers = len(self.channels)
        if self.kernel_size != DefaultFromLibrary.YES:
            _ = ensure_list_of_tuples(self.kernel_size, dim, n_layers, "kernel_size")
        if self.stride != DefaultFromLibrary.YES:
            _ = ensure_list_of_tuples(self.stride, dim, n_layers, "stride")
        if self.padding != DefaultFromLibrary.YES:
            _ = ensure_list_of_tuples(self.padding, dim, n_layers, "padding")
        if self.dilation != DefaultFromLibrary.YES:
            _ = ensure_list_of_tuples(self.dilation, dim, n_layers, "dilation")

    def check_pool_indices(self, indices: Optional[Sequence[int]]) -> Sequence[int]:
        return check_pool_indices(indices, n_layers=len(self.channels))


class ConvEncoderOptions(_ConvOptions):
    """
    Config class for ConvEncoder when it is a submodule.
    See for example: :py:class:`clinicadl.monai_networks.nn.cnn.CNN`
    """

    pooling: Union[
        Optional[PoolingParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    pooling_indices: Union[
        Optional[Sequence[int]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @model_validator(mode="after")
    def check_pooling(self):
        if self.pooling_indices != DefaultFromLibrary.YES:
            checked_indices = self.check_pool_indices(self.pooling_indices)
            if self.pooling != DefaultFromLibrary.YES:
                _ = check_pool_layers(self.pooling, pooling_indices=checked_indices)

        return self


class ConvEncoderConfig(NetworkConfig, _FullyConvConfig, ConvEncoderOptions):
    """Config class for ConvEncoder."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.CONV_ENCODER

    @model_validator(mode="after")
    def check_dim(self):
        self.check_args_dim(self.spatial_dims)
        return self


class ConvDecoderOptions(_ConvOptions):
    """
    Config class for ConvDecoder when it is a submodule.
    See for example: :py:class:`clinicadl.monai_networks.nn.generator.Generator`
    """

    output_padding: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    unpooling: Union[
        Optional[UnpoolingParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    unpooling_indices: Union[
        Optional[Sequence[int]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @model_validator(mode="after")
    def check_unpooling(self):
        if self.unpooling_indices != DefaultFromLibrary.YES:
            checked_indices = self.check_pool_indices(self.unpooling_indices)
            if self.unpooling != DefaultFromLibrary.YES:
                _ = check_unpool_layers(
                    self.unpooling, unpooling_indices=checked_indices
                )

        return self

    def check_args_dim(self, dim: int) -> None:
        super().check_args_dim(dim)
        if self.output_padding != DefaultFromLibrary.YES:
            _ = ensure_list_of_tuples(
                self.output_padding, dim, len(self.channels), "output_padding"
            )


class ConvDecoderConfig(NetworkConfig, _FullyConvConfig, ConvDecoderOptions):
    """Config class for ConvDecoder."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.CONV_DECODER

    @model_validator(mode="after")
    def check_dim(self):
        self.check_args_dim(self.spatial_dims)
        return self
