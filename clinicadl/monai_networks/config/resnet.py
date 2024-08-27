from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple, Union

from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)

from clinicadl.utils.factories import DefaultFromLibrary

from .base import NetworkBaseConfig

__all__ = ["ResNetConfig", "ResNetFeaturesConfig", "SegResNetConfig"]


class ResNetBlocks(str, Enum):
    """Supported ResNet blocks."""

    BASIC = "basic"
    BOTTLENECK = "bottleneck"


class ShortcutTypes(str, Enum):
    """Supported shortcut types."""

    A = "A"
    B = "B"


class ResNets(str, Enum):
    """Supported ResNet networks."""

    RESNET_10 = "resnet10"
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"
    RESNET_200 = "resnet200"


class UpsampleModes(str, Enum):
    """Supported upsampling modes."""

    DECONV = "deconv"
    NON_TRAINABLE = "nontrainable"
    PIXEL_SHUFFLE = "pixelshuffle"


class ResNetConfig(NetworkBaseConfig):
    """Config class for ResNet."""

    block: ResNetBlocks
    layers: Tuple[PositiveInt, PositiveInt, PositiveInt, PositiveInt]
    block_inplanes: Tuple[PositiveInt, PositiveInt, PositiveInt, PositiveInt]

    spatial_dims: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    n_input_channels: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    conv1_t_size: Union[
        PositiveInt, Tuple[PositiveInt, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    conv1_t_stride: Union[
        PositiveInt, Tuple[PositiveInt, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    no_max_pool: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    shortcut_type: Union[ShortcutTypes, DefaultFromLibrary] = DefaultFromLibrary.YES
    widen_factor: Union[PositiveFloat, DefaultFromLibrary] = DefaultFromLibrary.YES
    num_classes: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    feed_forward: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    bias_downsample: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    def dim(self) -> int:
        """Dimension of the images."""
        return self.spatial_dims

    @model_validator(mode="after")
    def model_validator(self):
        """Checks coherence between parameters."""
        if self.conv1_t_size != DefaultFromLibrary.YES:
            assert self._check_dimensions(
                self.conv1_t_size
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for conv1_t_size. You passed {self.conv1_t_size}."
        if self.conv1_t_stride != DefaultFromLibrary.YES:
            assert self._check_dimensions(
                self.conv1_t_stride
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for conv1_t_stride. You passed {self.conv1_t_stride}."

        return self


class ResNetFeaturesConfig(NetworkBaseConfig):
    """Config class for ResNet backbones."""

    model_name: ResNets

    pretrained: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    spatial_dims: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    in_channels: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    def dim(self) -> int:
        """Dimension of the images."""
        return self.spatial_dims

    @model_validator(mode="after")
    def model_validator(self):
        """Checks coherence between parameters."""
        if self.pretrained == DefaultFromLibrary.YES or self.pretrained:
            assert (
                self.spatial_dims == DefaultFromLibrary.YES or self.spatial_dims == 3
            ), "Pretrained weights are only available with spatial_dims=3. Otherwise, set pretrained to False."
            assert (
                self.in_channels == DefaultFromLibrary.YES or self.in_channels == 1
            ), "Pretrained weights are only available with in_channels=1. Otherwise, set pretrained to False."

        return self


class SegResNetConfig(NetworkBaseConfig):
    """Config class for SegResNet."""

    spatial_dims: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    init_filters: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    in_channels: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    out_channels: Union[PositiveInt, DefaultFromLibrary] = DefaultFromLibrary.YES
    dropout_prob: Union[
        Optional[NonNegativeFloat], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    use_conv_final: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
    blocks_down: Union[
        Tuple[PositiveInt, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    blocks_up: Union[
        Tuple[PositiveInt, ...], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    upsample_mode: Union[UpsampleModes, DefaultFromLibrary] = DefaultFromLibrary.YES

    @computed_field
    def dim(self) -> int:
        """Dimension of the images."""
        return self.spatial_dims

    @field_validator("dropout_prob")
    @classmethod
    def validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        return cls.base_validator_dropout(v)
