from typing import Optional, Sequence, Union

from pydantic import PositiveInt, model_validator

import clinicadl.networks.nn as nets
from clinicadl.networks.nn.layers.utils import ActivationParameters
from clinicadl.networks.nn.resnet import ResNetBlockType
from clinicadl.networks.nn.senet import check_se_channels
from clinicadl.utils.config import update_kwargs_with_defaults
from clinicadl.utils.factories import DefaultFromLibrary

from .base import (
    NetworkConfig,
    _OptionalNumOutputsConfig,
    _OutputActConfig,
)
from .resnet import ResNetConfig

__all__ = [
    "SEResNetConfig",
    "SEResNet50Config",
    "SEResNet101Config",
    "SEResNet152Config",
]


class SEResNetConfig(ResNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet`.
    """

    se_reduction: PositiveInt

    def __init__(
        self,
        spatial_dims: PositiveInt,
        in_channels: PositiveInt,
        num_outputs: Optional[PositiveInt],
        se_reduction: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES,
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
        kwargs = locals()
        del kwargs["self"]
        kwargs = update_kwargs_with_defaults(kwargs, function=nets.SEResNet.__init__)
        kwargs = update_kwargs_with_defaults(kwargs, function=nets.ResNet.__init__)
        super(ResNetConfig, self).__init__(**kwargs)

    @model_validator(mode="after")
    def check_se_channels(self):
        check_se_channels(self.n_features, self.se_reduction)

        return self


class _FromLiteratureConfig(
    NetworkConfig,
    _OptionalNumOutputsConfig,
    _OutputActConfig,
):
    """Base config class for networks from literature."""

    def __init__(
        self,
        num_outputs: Optional[PositiveInt],
        output_act: Union[Optional[ActivationParameters], DefaultFromLibrary] = (
            DefaultFromLibrary.YES
        ),
    ):
        super().__init__(
            num_outputs=num_outputs,
            output_act=output_act,
        )


class SEResNet50Config(_FromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet50`.
    """


class SEResNet101Config(_FromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet101`.
    """


class SEResNet152Config(_FromLiteratureConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet152`.
    """
