from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.utils import get_act_layer

from .layers.unet import ConvBlock, DownBlock, UpSample
from .layers.utils import ActFunction, ActivationParameters


class BaseUNet(nn.Module, ABC):
    """Base class for UNet and AttentionUNet."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (64, 128, 256, 512, 1024),
        act: ActivationParameters = ActFunction.RELU,
        output_act: Optional[ActivationParameters] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        if not isinstance(channels, Sequence) or len(channels) < 2:
            raise ValueError(
                f"channels should be a sequence, whose length is no less than 2. Got {channels}"
            )
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.act = act
        self.output_act = output_act
        self.dropout = dropout

        self.doubleconv = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            act=act,
            dropout=dropout,
        )
        self._build_encoder()
        self._build_decoder()
        self.reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.output_act = get_act_layer(output_act) if output_act else None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _build_encoder(self) -> None:
        for i in range(1, len(self.channels)):
            self.add_module(
                f"down{i}",
                DownBlock(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.channels[i - 1],
                    out_channels=self.channels[i],
                    act=self.act,
                    dropout=self.dropout,
                ),
            )

    @abstractmethod
    def _build_decoder(self) -> None:
        pass


class UNet(BaseUNet):
    """
    UNet based on [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597).

    The user can customize the number of encoding blocks, the number of channels in each block, as well as other parameters
    like the activation function.

    .. warning:: UNet works only with images whose dimensions are high enough powers of 2. More precisely, if n is the number
    of max pooling operation in your UNet (which is equal to `len(channels)-1`), the image must have :math:`2^{k}`
    pixels in each dimension, with :math:`k \\geq n` (e.g. shape (:math:`2^{n}`, :math:`2^{n+3}`) for a 2D image).

    Note: the implementation proposed here is not exactly the one described in the original paper. Padding is added to
    convolutions so that the feature maps keep a constant size (except when they are passed to `max pool` or `up-sample`
    layers), batch normalization is used, and `up-conv` layers are here made with an [Upsample](https://pytorch.org/docs/
    stable/generated/torch.nn.Upsample.html) layer followed by a 3x3 convolution.

    Parameters
    ----------
    spatial_dims : int
        number of spatial dimensions of the input image.
    in_channels : int
        number of channels in the input image.
    out_channels : int
        number of output channels.
    channels : Sequence[int] (optional, default=(64, 128, 256, 512, 1024))
        sequence of integers stating the number of channels in each UNet block. Thus, this parameter also controls
        the number of UNet blocks. The length `channels` should be nos less than 2.\n
        Default to `(64, 128, 256, 512, 1024)`, as in the original paper.
    act : ActivationParameters (optional, default=ActFunction.RELU)
        the activation function used in the convolutional part, and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`.
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.\n
        Default is "relu", as in the original paper.
    output_act : Optional[ActivationParameters] (optional, default=None)
        a potential activation layer applied to the output of the network. Should be pass in the same way as `act`.
        If None, no last activation will be applied.
    dropout : Optional[float] (optional, default=None)
        dropout ratio. If None, no dropout.

    Examples
    --------
    >>> UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(4, 8),
            act="elu",
            output_act=("softmax", {"dim": 1}),
            dropout=0.1,
        )
    UNet(
        (doubleconv): ConvBlock(
            (0): Convolution(
                (conv): Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (adn): ADN(
                    (N): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (D): Dropout(p=0.1, inplace=False)
                    (A): ELU(alpha=1.0)
                )
            )
            (1): Convolution(
                (conv): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (adn): ADN(
                    (N): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (D): Dropout(p=0.1, inplace=False)
                    (A): ELU(alpha=1.0)
                )
            )
        )
        (down1): DownBlock(
            (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (doubleconv): ConvBlock(
                (0): Convolution(
                    (conv): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (adn): ADN(
                        (N): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (D): Dropout(p=0.1, inplace=False)
                        (A): ELU(alpha=1.0)
                    )
                )
                (1): Convolution(
                    (conv): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (adn): ADN(
                        (N): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (D): Dropout(p=0.1, inplace=False)
                        (A): ELU(alpha=1.0)
                    )
                )
            )
        )
        (upsample1): UpSample(
            (0): Upsample(scale_factor=2.0, mode='nearest')
            (1): Convolution(
                (conv): Conv2d(8, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (adn): ADN(
                    (N): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (D): Dropout(p=0.1, inplace=False)
                    (A): ELU(alpha=1.0)
                )
            )
        )
        (doubleconv1): ConvBlock(
            (0): Convolution(
                (conv): Conv2d(8, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (adn): ADN(
                    (N): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (D): Dropout(p=0.1, inplace=False)
                    (A): ELU(alpha=1.0)
                )
            )
            (1): Convolution(
                (conv): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (adn): ADN(
                    (N): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (D): Dropout(p=0.1, inplace=False)
                    (A): ELU(alpha=1.0)
                )
            )
        )
        (reduce_channels): Convolution(
            (conv): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
        )
        (output_act): Softmax(dim=1)
    )
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_history = [self.doubleconv(x)]

        for i in range(1, len(self.channels)):
            x = self.get_submodule(f"down{i}")(x_history[-1])
            x_history.append(x)

        x_history.pop()  # the output of bottelneck is not used as a residual
        for i in range(len(self.channels) - 1, 0, -1):
            up = self.get_submodule(f"upsample{i}")(x)
            merged = torch.cat((x_history.pop(), up), dim=1)
            x = self.get_submodule(f"doubleconv{i}")(merged)

        out = self.reduce_channels(x)

        if self.output_act:
            out = self.output_act(out)

        return out

    def _build_decoder(self):
        for i in range(len(self.channels) - 1, 0, -1):
            self.add_module(
                f"upsample{i}",
                UpSample(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.channels[i],
                    out_channels=self.channels[i - 1],
                    act=self.act,
                    dropout=self.dropout,
                ),
            )
            self.add_module(
                f"doubleconv{i}",
                ConvBlock(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.channels[i - 1] * 2,
                    out_channels=self.channels[i - 1],
                    act=self.act,
                    dropout=self.dropout,
                ),
            )
