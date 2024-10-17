from typing import Optional, Sequence, Union

from pydantic import PositiveInt, computed_field

from clinicadl.monai_networks.nn.layers.utils import ActivationParameters
from clinicadl.monai_networks.nn.resnet import ResNetBlockType
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetworks, NetworkConfig, NetworkType, PreTrainedConfig


class ResNetConfig(NetworkConfig):
    """Config class for ResNet."""

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    num_outputs: Optional[PositiveInt]
    block_type: Union[str, ResNetBlockType, DefaultFromLibrary] = DefaultFromLibrary.YES
    n_res_blocks: Union[
        Sequence[PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    n_features: Union[
        Sequence[PositiveInt], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    init_conv_size: Union[
        Sequence[PositiveInt], PositiveInt, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    init_conv_stride: Union[
        Sequence[PositiveInt], PositiveInt, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    bottleneck_reduction: Union[
        PositiveInt, DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    act: Union[ActivationParameters, DefaultFromLibrary] = DefaultFromLibrary.YES
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.RESNET


class PreTrainedResNetConfig(PreTrainedConfig):
    """Base config class for SOTA ResNets."""

    @computed_field
    @property
    def _type(self) -> NetworkType:
        """To know where to look for the network."""
        return NetworkType.RESNET


class ResNet18Config(PreTrainedResNetConfig):
    """Config class for ResNet-18."""

    @computed_field
    @property
    def name(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.RESNET_18


class ResNet34Config(PreTrainedResNetConfig):
    """Config class for ResNet-34."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.RESNET_34


class ResNet50Config(PreTrainedResNetConfig):
    """Config class for ResNet-50."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.RESNET_50


class ResNet101Config(PreTrainedResNetConfig):
    """Config class for ResNet-101."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.RESNET_101


class ResNet152Config(PreTrainedResNetConfig):
    """Config class for ResNet-152."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.RESNET_152
