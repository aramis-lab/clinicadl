from typing import Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, computed_field

from clinicadl.monai_networks.nn.layers.utils import (
    ActivationParameters,
    ConvNormalizationParameters,
    ConvParameters,
    UnpoolingParameters,
)
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetworks, NetworkConfig


class ConvDecoderOptions(BaseModel):
    """
    Config class for ConvDecoder when it is a submodule.
    See for example: :py:class:`clinicadl.monai_networks.nn.generator.Generator`
    """

    channels: Sequence[PositiveInt]
    kernel_size: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    stride: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    padding: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    output_padding: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    dilation: Union[ConvParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    unpooling: Union[
        Optional[UnpoolingParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    unpooling_indices: Union[
        Optional[Sequence[int]], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    norm: Union[
        Optional[ConvNormalizationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    dropout: Union[Optional[PositiveFloat], DefaultFromLibrary] = DefaultFromLibrary.YES
    bias: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    adn_ordering: Union[str, DefaultFromLibrary] = DefaultFromLibrary.YES

    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
    )


class ConvDecoderConfig(NetworkConfig, ConvDecoderOptions):
    """Config class for ConvDecoder."""

    spatial_dims: PositiveInt
    in_channels: PositiveInt

    @computed_field
    @property
    def name(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.CONV_DECODER
