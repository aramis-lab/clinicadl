from typing import Any, Optional

import torch.nn as nn
from pydantic import PositiveInt, model_validator

from clinicadl.utils.factories import get_defaults_from

from .layers.utils import ActivationParameters
from .resnet import GeneralResNet, ResNet, ResNetBlockType, ResNetConfig
from .utils.config import NetworkConfig

__all__ = ["SEResNet", "SEResNet50", "SEResNet101", "SEResNet152"]


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
        config = SEResNetConfig(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_outputs=num_outputs,
            se_reduction=se_reduction,
            **kwargs,
        )
        super().__init__(
            **config.to_raw_dict(),
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
        config = SEResNet50Config(num_outputs=num_outputs, output_act=output_act)
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=config.num_outputs,
            n_res_blocks=(3, 4, 6, 3),
            block_type=ResNetBlockType.BOTTLENECK,
            n_features=(256, 512, 1024, 2048),
            output_act=config.output_act,
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
        config = SEResNet101Config(num_outputs=num_outputs, output_act=output_act)
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=config.num_outputs,
            n_res_blocks=(3, 4, 23, 3),
            block_type=ResNetBlockType.BOTTLENECK,
            n_features=(256, 512, 1024, 2048),
            output_act=config.output_act,
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
        config = SEResNet152Config(num_outputs=num_outputs, output_act=output_act)
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=config.num_outputs,
            n_res_blocks=(3, 8, 36, 3),
            block_type=ResNetBlockType.BOTTLENECK,
            n_features=(256, 512, 1024, 2048),
            output_act=config.output_act,
        )


SE_RES_NET_DEFAULTS = get_defaults_from(SEResNet)
SE_RES_NET_50_DEFAULTS = get_defaults_from(SEResNet50)
SE_RES_NET_101_DEFAULTS = get_defaults_from(SEResNet101)
SE_RES_NET_152_DEFAULTS = get_defaults_from(SEResNet152)


class SEResNetConfig(ResNetConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet`.
    """

    se_reduction: PositiveInt = SE_RES_NET_DEFAULTS["se_reduction"]

    @model_validator(mode="after")
    def _check_se_channels(self):
        for n in self.n_features:
            if n < self.se_reduction:
                raise ValueError(
                    f"elements of n_features must be greater or equal to se_reduction. Got {n} in n_features "
                    f"and se_reduction={self.se_reduction}"
                )

        return self

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return SEResNet


class SEResNet50Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet50`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = SE_RES_NET_50_DEFAULTS["output_act"]

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return SEResNet50


class SEResNet101Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet101`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = SE_RES_NET_101_DEFAULTS["output_act"]

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return SEResNet101


class SEResNet152Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.SEResNet152`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = SE_RES_NET_152_DEFAULTS["output_act"]

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return SEResNet152
