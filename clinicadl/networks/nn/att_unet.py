import torch.nn as nn

from .layers.unet import AttentionUpBlock
from .unet import UNet


class AttentionUNet(UNet):
    """
    Attention-UNet, based on `Attention U-Net: Learning Where to Look for the Pancreas <https://arxiv.org/abs/1804.03999>`_.

    Very similar to :py:class:`~clinicadl.networks.nn.UNet`, but with attention gates in the skip connections.

    The user can customize the number of encoding blocks, the number of channels in each block, as well as other parameters
    like the activation function.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    .. warning::
        ``AttentionUNet`` works only with images whose dimensions are high enough powers of 2. More precisely, if ``n`` is the number
        of max pooling operation in your ``UNet`` (which is equal to ``len(channels)-1``), the image must have :math:`2^{k}`
        pixels in each dimension, with :math:`k \\geq n` (e.g. shape (:math:`2^{n}`, :math:`2^{n+3}`, :math:`2^{n+1}`) for a 3D image).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of output channels.
    channels : Sequence[int], default=(64, 128, 256, 512, 1024)
        Number of channels in each UNet block. Thus, this parameter also controls
        the number of UNet blocks (equal to the length of the sequence). The length ``channels`` should be no less than ``2``.\n
        Default to ``(64, 128, 256, 512, 1024)``, as in the original UNet paper\\ :footcite:p:`Ronneberger2015`.
    act : ActivationParameters, default="relu"
        The activation function used, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.\n
        Default is ``relu``, as in the original paper.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network. Must be passed in the same way as ``act``.
        If ``None``, no last activation will be applied.
    dropout : Optional[float], default=None
        Dropout ratio. If ``None``, no dropout.

    Examples
    --------

    .. code-block:: python

        # an AttentionUNet with 1 downsampling
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
            (up1): AttentionUpBlock(
                (upsample): UpSample(
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
                (attention): AttentionBlock(
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
                (doubleconv): ConvBlock(
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
            )
            (reduce_channels): Convolution(
                (conv): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
            )
            (output_act): Softmax(dim=1)
        )

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.UNet`

    References
    ----------
    .. footbibliography::

    """

    @property
    def _decoding_block(self) -> type[nn.Module]:
        return AttentionUpBlock
