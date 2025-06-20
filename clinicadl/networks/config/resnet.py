from typing import Optional, Sequence, Union

from pydantic import PositiveInt, model_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.networks.nn.resnet import (
    ResNetBlockType,
    bottleneck_reduce,
    check_res_blocks,
)
from clinicadl.networks.nn.utils import ensure_tuple
from clinicadl.utils.factories import get_defaults_from

from .base import (
    NetworkConfig,
    _SpatialDimsConfig,
)

__all__ = [
    "ResNetConfig",
    "ResNet18Config",
    "ResNet34Config",
    "ResNet50Config",
    "ResNet101Config",
    "ResNet152Config",
]

RES_NET_DEFAULTS = get_defaults_from(nets.ResNet)
RES_NET_18_DEFAULTS = get_defaults_from(nets.ResNet18)
RES_NET_34_DEFAULTS = get_defaults_from(nets.ResNet34)
RES_NET_50_DEFAULTS = get_defaults_from(nets.ResNet50)
RES_NET_101_DEFAULTS = get_defaults_from(nets.ResNet101)
RES_NET_152_DEFAULTS = get_defaults_from(nets.ResNet152)


class ResNetConfig(NetworkConfig, _SpatialDimsConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    num_outputs: Optional[PositiveInt]
    block_type: ResNetBlockType = RES_NET_DEFAULTS["block_type"]
    n_res_blocks: Sequence[PositiveInt] = RES_NET_DEFAULTS["n_res_blocks"]
    n_features: Sequence[PositiveInt] = RES_NET_DEFAULTS["n_features"]
    init_conv_size: Union[Sequence[PositiveInt], PositiveInt] = RES_NET_DEFAULTS[
        "init_conv_size"
    ]
    init_conv_stride: Union[Sequence[PositiveInt], PositiveInt] = RES_NET_DEFAULTS[
        "init_conv_stride"
    ]
    bottleneck_reduction: PositiveInt = RES_NET_DEFAULTS["bottleneck_reduction"]
    act: Optional[ActivationParameters] = RES_NET_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = RES_NET_DEFAULTS["output_act"]

    @model_validator(mode="after")
    def make_checks(self):
        check_res_blocks(self.n_res_blocks, self.n_features)
        bottleneck_reduce(self.n_features, self.bottleneck_reduction)
        ensure_tuple(self.init_conv_size, self.spatial_dims, "init_conv_size")
        ensure_tuple(self.init_conv_stride, self.spatial_dims, "init_conv_stride")

        return self


class ResNet18Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet18`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = RES_NET_18_DEFAULTS["output_act"]
    pretrained: bool = RES_NET_18_DEFAULTS["pretrained"]


class ResNet34Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet34`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = RES_NET_34_DEFAULTS["output_act"]
    pretrained: bool = RES_NET_34_DEFAULTS["pretrained"]


class ResNet50Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet50`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = RES_NET_50_DEFAULTS["output_act"]
    pretrained: bool = RES_NET_50_DEFAULTS["pretrained"]


class ResNet101Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet101`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = RES_NET_101_DEFAULTS["output_act"]
    pretrained: bool = RES_NET_101_DEFAULTS["pretrained"]


class ResNet152Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet152`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = RES_NET_152_DEFAULTS["output_act"]
    pretrained: bool = RES_NET_152_DEFAULTS["pretrained"]
