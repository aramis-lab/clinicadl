from typing import Optional, Union

from pydantic import PositiveInt, computed_field

from clinicadl.monai_networks.nn.layers.utils import ActivationParameters
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetworks, NetworkConfig
from .resnet import ResNetConfig


class SEResNetConfig(ResNetConfig):
    """Config class for Squeeze-and-Excitation ResNet."""

    se_reduction: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.SE_RESNET


class SEResNetPreTrainedConfig(NetworkConfig):
    """Base config class for SOTA SE-ResNets."""

    num_outputs: Optional[PositiveInt]
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES


class SEResNet50Config(SEResNetPreTrainedConfig):
    """Config class for SE-ResNet-50."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.SE_RESNET_50


class SEResNet101Config(SEResNetPreTrainedConfig):
    """Config class for SE-ResNet-101."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.SE_RESNET_101


class SEResNet152Config(SEResNetPreTrainedConfig):
    """Config class for SE-ResNet-152."""

    @computed_field
    @property
    def network(self) -> ImplementedNetworks:
        """The name of the network."""
        return ImplementedNetworks.SE_RESNET_152
