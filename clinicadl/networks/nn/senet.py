from typing import Any, Optional, Sequence

from clinicadl.utils.factories import get_defaults_from

from .layers.utils import ActivationParameters
from .resnet import GeneralResNet, ResNet, ResNetBlockType

__all__ = ["SEResNet", "SEResNet50", "SEResNet101", "SEResNet152", "check_se_channels"]


class SEResNet(GeneralResNet):
    """
    Squeeze-and-Excitation ResNet, based on `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_.

    ``SEResNet`` is very similar to :py:class:`~clinicadl.networks.nn.ResNet`, except that
    Squeeze-and-Excitation blocks are added before residual connections.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of channels in the input image.
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.
        If ``None``, the feature map before the last fully connected layer will be returned.
    se_reduction : int, default=16
        Reduction ratio in the bottelneck layer of the excitation modules. Default to ``16``, as in the original
        paper.
    kwargs : Any
        Any optional argument accepted by :py:class:`~clinicadl.networks.nn.ResNet`.

    Examples
    --------

    .. code-block:: python

        >>> SEResNet(
                spatial_dims=2,
                in_channels=1,
                num_outputs=2,
                block_type="basic",
                se_reduction=2,
                n_features=(8,),
                n_res_blocks=(2,),
                output_act="softmax",
                init_conv_size=5,
            )
        SEResNet(
            (conv0): Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
            (norm0): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act0): ReLU(inplace=True)
            (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            (layer1): Sequential(
                (0): SEResNetBlock(
                    (conv1): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (norm1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act1): ReLU(inplace=True)
                    (conv2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (norm2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (se_layer): ChannelSELayer(
                        (avg_pool): AdaptiveAvgPool2d(output_size=1)
                        (fc): Sequential(
                            (0): Linear(in_features=8, out_features=4, bias=True)
                            (1): ReLU(inplace=True)
                            (2): Linear(in_features=4, out_features=8, bias=True)
                            (3): Sigmoid()
                        )
                    )
                    (act2): ReLU(inplace=True)
                )
                (1): SEResNetBlock(
                    (conv1): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (norm1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act1): ReLU(inplace=True)
                    (conv2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (norm2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (se_layer): ChannelSELayer(
                        (avg_pool): AdaptiveAvgPool2d(output_size=1)
                        (fc): Sequential(
                            (0): Linear(in_features=8, out_features=4, bias=True)
                            (1): ReLU(inplace=True)
                            (2): Linear(in_features=4, out_features=8, bias=True)
                            (3): Sigmoid()
                        )
                    )
                    (act2): ReLU(inplace=True)
                )
            )
            (fc): Sequential(
                (pool): AdaptiveAvgPool2d(output_size=(1, 1))
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (out): Linear(in_features=8, out_features=2, bias=True)
                (output_act): Softmax(dim=None)
            )
        )

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ResNet`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_outputs: Optional[int],
        se_reduction: int = 16,
        **kwargs: Any,
    ) -> None:
        # get defaults from resnet
        default_resnet_args = get_defaults_from(ResNet.__init__)
        for arg, value in default_resnet_args.items():
            if arg not in kwargs:
                kwargs[arg] = value

        check_se_channels(kwargs["n_features"], se_reduction)

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_outputs=num_outputs,
            se_reduction=se_reduction,
            **kwargs,
        )


class SEResNet50(ResNet):
    """
    SEResNet-50, from `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    .. warning:: Only works with **2D images with 3 channels**.

    Parameters
    ----------
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.
        If ``None``, the feature map before the last fully connected layer will be returned.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.SEResNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
    ) -> None:
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=num_outputs,
            n_res_blocks=(3, 4, 6, 3),
            block_type=ResNetBlockType.BOTTLENECK,
            n_features=(256, 512, 1024, 2048),
            output_act=output_act,
        )


class SEResNet101(ResNet):
    """
    SEResNet-101, from `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    .. warning:: Only works with **2D images with 3 channels**.

    Parameters
    ----------
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.
        If ``None``, the feature map before the last fully connected layer will be returned.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.SEResNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
    ) -> None:
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=num_outputs,
            n_res_blocks=(3, 4, 23, 3),
            block_type=ResNetBlockType.BOTTLENECK,
            n_features=(256, 512, 1024, 2048),
            output_act=output_act,
        )


class SEResNet152(ResNet):
    """
    SEResNet-152, from `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    .. warning:: Only works with **2D images with 3 channels**.

    Parameters
    ----------
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.
        If ``None``, the feature map before the last fully connected layer will be returned.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.SEResNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
    ) -> None:
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=num_outputs,
            n_res_blocks=(3, 8, 36, 3),
            block_type=ResNetBlockType.BOTTLENECK,
            n_features=(256, 512, 1024, 2048),
            output_act=output_act,
        )


def check_se_channels(n_features: Sequence[int], se_reduction: int) -> None:
    """
    Checks that the output of residual blocks always have a number of channels greater
    than squeeze-excitation bottleneck reduction factor.
    """
    for n in n_features:
        if n < se_reduction:
            raise ValueError(
                f"elements of n_features must be greater or equal to se_reduction. Got {n} in n_features "
                f"and se_reduction={se_reduction}"
            )
