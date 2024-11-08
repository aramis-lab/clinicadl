from typing import Union

from pydantic import PositiveInt, computed_field, model_validator

from clinicadl.networks.nn.senet import check_se_channels
from clinicadl.utils.factories import DefaultFromLibrary

from .base import ImplementedNetwork, NetworkType, _PreTrainedConfig
from .resnet import ResNetConfig

__all__ = [
    "SEResNetConfig",
    "SEResNet50Config",
    "SEResNet101Config",
    "SEResNet152Config",
]


class SEResNetConfig(ResNetConfig):
    """Config class for Squeeze-and-Excitation ResNet."""

    se_reduction: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.SE_RESNET

    @model_validator(mode="after")
    def check_se_channels(self):
        if (
            self.n_features != DefaultFromLibrary.YES
            and self.se_reduction != DefaultFromLibrary.YES
        ):
            check_se_channels(self.n_features, self.se_reduction)

        return self


class _PreTrainedSEResNetConfig(_PreTrainedConfig):
    """Base config class for SOTA SE-ResNets."""

    @computed_field
    @property
    def _type(self) -> NetworkType:
        """To know where to look for the network."""
        return NetworkType.SE_RESNET


class SEResNet50Config(_PreTrainedSEResNetConfig):
    """Config class for SE-ResNet-50."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.SE_RESNET_50


class SEResNet101Config(_PreTrainedSEResNetConfig):
    """Config class for SE-ResNet-101."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.SE_RESNET_101


class SEResNet152Config(_PreTrainedSEResNetConfig):
    """Config class for SE-ResNet-152."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.SE_RESNET_152
