from typing import Optional

import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.utils import get_pool_layer

from .utils import ActFunction, ActivationParameters, NormLayer


class ConvBlock(nn.Sequential):
    """UNet doouble convolution block."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Optional[ActivationParameters] = ActFunction.RELU,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.add_module(
            "0",
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act=act,
                norm=NormLayer.BATCH,
                dropout=dropout,
            ),
        )
        self.add_module(
            "1",
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act=act,
                norm=NormLayer.BATCH,
                dropout=dropout,
            ),
        )


class UpSample(nn.Sequential):
    """UNet up-conv block with first upsampling and then a convolution."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Optional[ActivationParameters] = ActFunction.RELU,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.add_module("0", nn.Upsample(scale_factor=2))
        self.add_module(
            "1",
            Convolution(
                spatial_dims,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=3,
                act=act,
                adn_ordering="NDA",
                norm=NormLayer.BATCH,
                dropout=dropout,
            ),
        )


class DownBlock(nn.Sequential):
    """UNet down block with first max pooling and then two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Optional[ActivationParameters] = ActFunction.RELU,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.pool = get_pool_layer(("max", {"kernel_size": 2}), spatial_dims)
        self.doubleconv = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            act=act,
            dropout=dropout,
        )
