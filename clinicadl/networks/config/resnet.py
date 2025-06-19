from typing import Optional, Sequence, Union

from pydantic import PositiveInt, model_validator

from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.networks.nn.resnet import (
    ResNetBlockType,
    bottleneck_reduce,
    check_res_blocks,
)
from clinicadl.networks.nn.utils import ensure_tuple
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    NetworkConfig,
    _FullyConvConfig,
    _MandatoryActConfig,
    _OptionalNumOutputsConfig,
    _OutputActConfig,
    _PretrainedFromLiteratureConfig,
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
    _OptionalNumOutputsConfig,
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


class ResNet18Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet18`.
    """


class ResNet34Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet34`.
    """


class ResNet50Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet50`.
    """


class ResNet101Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet101`.
    """


class ResNet152Config(_PretrainedFromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ResNet152`.
    """
