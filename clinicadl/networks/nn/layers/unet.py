from typing import Optional

import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.utils import get_pool_layer
from monai.networks.nets.attentionunet import AttentionBlock

from .utils import ActFunction, ActivationParameters, NormLayer


class ConvBlock(nn.Sequential):
    """UNet doouble convolution block."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: ActivationParameters = ActFunction.RELU,
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
        act: ActivationParameters = ActFunction.RELU,
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
        act: ActivationParameters = ActFunction.RELU,
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


class UpBlock(nn.Module):
    """UNet up block with upsampling, concatenation with skip connection,
    and two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: ActivationParameters = ActFunction.RELU,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.upsample = UpSample(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            act=act,
            dropout=dropout,
        )
        self.doubleconv = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels * 2,
            out_channels=out_channels,
            act=act,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """'skip' is from the skip connection."""
        up = self.upsample(x)
        merged = torch.cat((skip, up), dim=1)

        return self.doubleconv(merged)


class AttentionUpBlock(nn.Module):
    """AttentionUNet up block with upsampling, concatenation with attention skip connection,
    and two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: ActivationParameters = ActFunction.RELU,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.upsample = UpSample(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            act=act,
            dropout=dropout,
        )
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims,
            f_l=out_channels,
            f_g=out_channels,
            f_int=out_channels // 2,
            dropout=dropout,
        )
        self.doubleconv = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels * 2,
            out_channels=out_channels,
            act=act,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """'skip' is from the skip connection."""
        up = self.upsample(x)
        attentioned_skip = self.attention(g=skip, x=up)
        merged = torch.cat((attentioned_skip, up), dim=1)

        return self.doubleconv(merged)
