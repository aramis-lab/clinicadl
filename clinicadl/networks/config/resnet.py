from typing import Sequence, Union

from pydantic import PositiveInt, computed_field, model_validator

from clinicadl.networks.nn.resnet import (
    ResNetBlockType,
    bottleneck_reduce,
    check_res_blocks,
)
from clinicadl.networks.nn.utils import ensure_tuple
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    ImplementedNetwork,
    NetworkConfig,
    NetworkType,
    _FullyConvConfig,
    _MandatoryActConfig,
    _OptionalLastLinearLayersConfig,
    _OutputActConfig,
    _PreTrainedConfig,
)

__all__ = [
    "ResNetConfig",
    "ResNet18Config",
    "ResNet34Config",
    "ResNet50Config",
    "ResNet101Config",
    "ResNet152Config",
]


class ResNetConfig(
    NetworkConfig,
    _FullyConvConfig,
    _OptionalLastLinearLayersConfig,
    _MandatoryActConfig,
    _OutputActConfig,
):
    """Config class for ResNet."""

    block_type: Union[ResNetBlockType, DefaultFromLibrary] = DefaultFromLibrary.YES
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

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.RESNET

    @model_validator(mode="after")
    def make_checks(self):
        if self.n_features != DefaultFromLibrary.YES:
            if self.n_res_blocks != DefaultFromLibrary.YES:
                check_res_blocks(self.n_res_blocks, self.n_features)
            if self.bottleneck_reduction != DefaultFromLibrary.YES:
                _ = bottleneck_reduce(self.n_features, self.bottleneck_reduction)
        if self.init_conv_size != DefaultFromLibrary.YES:
            _ = ensure_tuple(self.init_conv_size, self.spatial_dims, "init_conv_size")
        if self.init_conv_stride != DefaultFromLibrary.YES:
            _ = ensure_tuple(
                self.init_conv_stride, self.spatial_dims, "init_conv_stride"
            )

        return self


class _PreTrainedResNetConfig(_PreTrainedConfig):
    """Base config class for SOTA ResNets."""

    @computed_field
    @property
    def _type(self) -> NetworkType:
        """To know where to look for the network."""
        return NetworkType.RESNET


class ResNet18Config(_PreTrainedResNetConfig):
    """Config class for ResNet-18."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.RESNET_18


class ResNet34Config(_PreTrainedResNetConfig):
    """Config class for ResNet-34."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.RESNET_34


class ResNet50Config(_PreTrainedResNetConfig):
    """Config class for ResNet-50."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.RESNET_50


class ResNet101Config(_PreTrainedResNetConfig):
    """Config class for ResNet-101."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.RESNET_101


class ResNet152Config(_PreTrainedResNetConfig):
    """Config class for ResNet-152."""

    @computed_field
    @property
    def name(self) -> ImplementedNetwork:
        """The name of the network."""
        return ImplementedNetwork.RESNET_152
