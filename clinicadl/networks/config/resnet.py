from typing import Any, Callable, Optional, Sequence, Union

import torch.nn as nn
from pydantic import PositiveInt, model_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
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
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet`.
    """

    block_type: ResNetBlockType
    n_res_blocks: Sequence[PositiveInt]
    n_features: Sequence[PositiveInt]
    init_conv_size: Union[Sequence[PositiveInt], PositiveInt]
    init_conv_stride: Union[Sequence[PositiveInt], PositiveInt]
    bottleneck_reduction: PositiveInt

    def __init__(
        self,
        spatial_dims: PositiveInt,
        in_channels: PositiveInt,
        num_outputs: Optional[PositiveInt],
        block_type: Union[ResNetBlockType, DefaultFromLibrary] = DefaultFromLibrary.YES,
        n_res_blocks: Union[Sequence[PositiveInt], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        n_features: Union[Sequence[PositiveInt], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        init_conv_size: Union[
            Sequence[PositiveInt], PositiveInt, DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        init_conv_stride: Union[
            Sequence[PositiveInt], PositiveInt, DefaultFromLibrary
        ] = (DefaultFromLibrary.YES),
        bottleneck_reduction: Union[PositiveInt, DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_outputs=num_outputs,
            block_type=block_type,
            n_res_blocks=n_res_blocks,
            n_features=n_features,
            init_conv_size=init_conv_size,
            init_conv_stride=init_conv_stride,
            bottleneck_reduction=bottleneck_reduction,
            act=act,
            output_act=output_act,
        )

    @model_validator(mode="after")
    def make_checks(self):
        check_res_blocks(self.n_res_blocks, self.n_features)
        bottleneck_reduce(self.n_features, self.bottleneck_reduction)
        ensure_tuple(self.init_conv_size, self.spatial_dims, "init_conv_size")
        ensure_tuple(self.init_conv_stride, self.spatial_dims, "init_conv_stride")

        return self


class _PreTrainedResNetConfig(_PreTrainedConfig):
    """Base config class for SOTA ResNets."""

    @classmethod
    def _get_class(cls) -> Callable[[Any], nn.Module]:
        """Returns the network associated to this config class."""
        return nets.get_resnet


class ResNet18Config(_PreTrainedResNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet18`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.RESNET_18.value


class ResNet34Config(_PreTrainedResNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet34`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.RESNET_34.value


class ResNet50Config(_PreTrainedResNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet50`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.RESNET_50.value


class ResNet101Config(_PreTrainedResNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet101`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.RESNET_101.value


class ResNet152Config(_PreTrainedResNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet152`.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return ImplementedNetwork.RESNET_152.value
