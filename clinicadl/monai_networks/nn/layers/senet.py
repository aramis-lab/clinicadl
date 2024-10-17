from typing import Callable, Optional

import torch
import torch.nn as nn
from monai.networks.blocks.squeeze_and_excitation import ChannelSELayer
from monai.networks.layers.factories import Conv, Norm
from monai.networks.layers.utils import get_act_layer

from .utils import ActivationParameters


class SEResNetBlock(nn.Module):
    """
    ResNet basic block. Adapted from MONAI's ResNetBlock:
    https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/
    monai/networks/nets/resnet.py#L71
    """

    expansion = 1
    reduction = 16

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        act: ActivationParameters = ("relu", {"inplace": True}),
    ) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(  # pylint: disable=not-callable
            in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False
        )
        self.norm1 = norm_type(planes)  # pylint: disable=not-callable
        self.act1 = get_act_layer(name=act)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)  # pylint: disable=not-callable
        self.norm2 = norm_type(planes)  # pylint: disable=not-callable
        self.se_layer = ChannelSELayer(
            spatial_dims=spatial_dims,
            in_channels=planes,
            r=self.reduction,
            acti_type_1=("relu", {"inplace": True}),
            acti_type_2="sigmoid",
        )
        self.downsample = downsample
        self.act2 = get_act_layer(name=act)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_layer(out)
        out += residual
        out = self.act2(out)

        return out


class SEResNetBottleneck(nn.Module):
    """
    ResNet bottleneck block. Adapted from MONAI's ResNetBottleneck:
    https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/
    monai/networks/nets/resnet.py#L124
    """

    expansion = 4
    reduction = 16

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        act: ActivationParameters = ("relu", {"inplace": True}),
    ) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)  # pylint: disable=not-callable
        self.norm1 = norm_type(planes)  # pylint: disable=not-callable
        self.act1 = get_act_layer(name=act)
        self.conv2 = conv_type(  # pylint: disable=not-callable
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm2 = norm_type(planes)  # pylint: disable=not-callable
        self.act2 = get_act_layer(name=act)
        self.conv3 = conv_type(  # pylint: disable=not-callable
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.norm3 = norm_type(planes * self.expansion)  # pylint: disable=not-callable
        self.se_layer = ChannelSELayer(
            spatial_dims=spatial_dims,
            in_channels=planes * self.expansion,
            r=self.reduction,
            acti_type_1=("relu", {"inplace": True}),
            acti_type_2="sigmoid",
        )
        self.downsample = downsample
        self.act3 = get_act_layer(name=act)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_layer(out)
        out += residual
        out = self.act3(out)

        return out
