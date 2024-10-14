from typing import Any

import torch
from monai.networks.nets.attentionunet import AttentionBlock

from .layers.unet import ConvBlock, UpSample
from .unet import BaseUNet


class AttentionUNet(BaseUNet):
    """
    Attention-UNet based on [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999).

    The user can customize the number of encoding blocks, the number of channels in each block, as well as other parameters
    like the activation function.

    .. warning:: AttentionUNet works only with images whose dimensions are high enough powers of 2. More precisely, if n is the
    number of max pooling operation in your AttentionUNet (which is equal to `len(channels)-1`), the image must have :math:`2^{k}`
    pixels in each dimension, with :math:`k \\geq n` (e.g. shape (:math:`2^{n}`, :math:`2^{n+3}`) for a 2D image).

    Parameters
    ----------
    spatial_dims : int
        number of spatial dimensions of the input image.
    in_channels : int
        number of channels in the input image.
    out_channels : int
        number of output channels.
    kwargs : Any
        any optional argument accepted by (:py:class:`clinicadl.monai_networks.nn.unet.UNet`).

    Examples
    --------
    >>> AttentionUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(4, 8),
            act="elu",
            output_act=("softmax", {"dim": 1}),
            dropout=0.1,
        )
    AttentionUNet(
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
        (attention1): AttentionBlock(
            (W_g): Sequential(
                (0): Convolution(
                    (conv): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
                )
                (1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (W_x): Sequential(
                (0): Convolution(
                    (conv): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
                )
                (1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (psi): Sequential(
                (0): Convolution(
                    (conv): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
                )
                (1): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): Sigmoid()
            )
            (relu): ReLU()
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

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )

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
                f"attention{i}",
                AttentionBlock(
                    spatial_dims=self.spatial_dims,
                    f_l=self.channels[i - 1],
                    f_g=self.channels[i - 1],
                    f_int=self.channels[i - 1] // 2,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_history = [self.doubleconv(x)]

        for i in range(1, len(self.channels)):
            x = self.get_submodule(f"down{i}")(x_history[-1])
            x_history.append(x)

        x_history.pop()  # the output of bottelneck is not used as a gating signal
        for i in range(len(self.channels) - 1, 0, -1):
            up = self.get_submodule(f"upsample{i}")(x)
            att_res = self.get_submodule(f"attention{i}")(g=x_history.pop(), x=up)
            merged = torch.cat((att_res, up), dim=1)
            x = self.get_submodule(f"doubleconv{i}")(merged)

        out = self.reduce_channels(x)

        if self.output_act:
            out = self.output_act(out)

        return out
