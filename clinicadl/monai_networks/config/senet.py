from typing import Union

from pydantic import PositiveInt, computed_field

from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetworks, NetworkType, PreTrainedConfig
from .resnet import ResNetConfig


class SEResNetConfig(ResNetConfig):
    """Config class for Squeeze-and-Excitation ResNet."""

    se_reduction: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.SE_RESNET


class PreTrainedSEResNetConfig(PreTrainedConfig):
    """Base config class for SOTA SE-ResNets."""

    @computed_field
    @property
    def _type(self) -> NetworkType:
        """To know where to look for the network."""
        return NetworkType.SE_RESNET


class SEResNet50Config(PreTrainedSEResNetConfig):
    """Config class for SE-ResNet-50."""

    @computed_field
    @property
    def name(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.SE_RESNET_50


class SEResNet101Config(PreTrainedSEResNetConfig):
    """Config class for SE-ResNet-101."""

    @computed_field
    @property
    def name(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.SE_RESNET_101


class SEResNet152Config(PreTrainedSEResNetConfig):
    """Config class for SE-ResNet-152."""

    @computed_field
    @property
    def name(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.SE_RESNET_152
