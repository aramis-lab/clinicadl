from typing import Callable, Optional

import torch
import torch.nn as nn
from monai.networks.blocks.squeeze_and_excitation import ChannelSELayer
from monai.networks.layers.factories import Conv, Norm
from monai.networks.layers.utils import get_act_layer

from clinicadl.monai_networks.nn.utils import ActivationParameters


class SEResNetBlock(nn.Module):
    expansion = 1
    reduction = 16

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        act: ActivationParameters = ("relu", {"inplace": True}),
    ) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(
            in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False
        )  # pylint: disable=not-callable
        self.bn1 = norm_type(planes)  # pylint: disable=not-callable
        self.act = get_act_layer(name=act)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)  # pylint: disable=not-callable
        self.bn2 = norm_type(planes)  # pylint: disable=not-callable
        self.se_layer = ChannelSELayer(
            spatial_dims=spatial_dims,
            in_channels=planes,
            r=self.reduction,
            acti_type_1=("relu", {"inplace": True}),
            acti_type_2="sigmoid",
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_layer(out)
        out += residual
        out = self.act(out)

        return out


class SEResNetBottleneck(nn.Module):
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
        self.bn1 = norm_type(planes)  # pylint: disable=not-callable
        self.conv2 = conv_type(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )  # pylint: disable=not-callable
        self.bn2 = norm_type(planes)  # pylint: disable=not-callable
        self.conv3 = conv_type(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )  # pylint: disable=not-callable
        self.bn3 = norm_type(planes * self.expansion)  # pylint: disable=not-callable
        self.se_layer = ChannelSELayer(
            spatial_dims=spatial_dims,
            in_channels=planes * self.expansion,
            r=self.reduction,
            acti_type_1=("relu", {"inplace": True}),
            acti_type_2="sigmoid",
        )
        self.act = get_act_layer(name=act)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_layer(out)
        out += residual
        out = self.act(out)

        return out
