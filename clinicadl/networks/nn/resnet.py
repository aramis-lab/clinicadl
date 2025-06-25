import re
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_act_layer
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from clinicadl.networks.nn.utils import ensure_tuple

from .layers.resnet import ResNetBlock, ResNetBottleneck
from .layers.senet import SEResNetBlock, SEResNetBottleneck
from .layers.utils import ActivationParameters

__all__ = [
    "GeneralResNet",
    "ResNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "bottleneck_reduce",
    "check_res_blocks",
]


class ResNetBlockType(str, Enum):
    """Supported ResNet blocks."""

    BASIC = "basic"
    BOTTLENECK = "bottleneck"


class GeneralResNet(nn.Module):
    """Common base class for ResNet and SEResNet."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_outputs: Optional[int],
        block_type: Union[str, ResNetBlockType],
        n_res_blocks: Sequence[int],
        n_features: Sequence[int],
        init_conv_size: Union[Sequence[int], int],
        init_conv_stride: Union[Sequence[int], int],
        bottleneck_reduction: int,
        se_reduction: Optional[int],
        act: ActivationParameters,
        output_act: ActivationParameters,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.block_type = block_type
        check_res_blocks(n_res_blocks, n_features)
        self.n_res_blocks = n_res_blocks
        self.n_features = n_features
        self.bottleneck_reduction = bottleneck_reduction
        self.se_reduction = se_reduction
        self.act = act
        self.squeeze_excitation = True if se_reduction else False

        self.init_conv_size = ensure_tuple(
            init_conv_size, spatial_dims, "init_conv_size"
        )
        self.init_conv_stride = ensure_tuple(
            init_conv_stride, spatial_dims, "init_conv_stride"
        )

        block, in_planes = self._get_block(block_type)

        conv_type, norm_type, pool_type, avgp_type = self._get_layers()

        block_avgpool = [0, 1, (1, 1), (1, 1, 1)]

        self.in_planes = in_planes[0]
        self.n_layers = len(in_planes)
        self.bias_downsample = False

        self.conv0 = conv_type(  # pylint: disable=not-callable
            in_channels,
            self.in_planes,
            kernel_size=self.init_conv_size,
            stride=self.init_conv_stride,
            padding=tuple(k // 2 for k in self.init_conv_size),
            bias=False,
        )
        self.norm0 = norm_type(self.in_planes)  # pylint: disable=not-callable
        self.act0 = get_act_layer(name=act)
        self.pool0 = pool_type(kernel_size=3, stride=2, padding=1)  # pylint: disable=not-callable
        self.layer1 = self._make_resnet_layer(
            block, in_planes[0], n_res_blocks[0], spatial_dims, act
        )
        for i, (n_blocks, n_feats) in enumerate(
            zip(n_res_blocks[1:], in_planes[1:]), start=2
        ):
            self.add_module(
                f"layer{i}",
                self._make_resnet_layer(
                    block,
                    planes=n_feats,
                    blocks=n_blocks,
                    spatial_dims=spatial_dims,
                    stride=2,
                    act=act,
                ),
            )
        self.fc = (
            nn.Sequential(
                OrderedDict(
                    [
                        ("pool", avgp_type(block_avgpool[spatial_dims])),  # pylint: disable=not-callable
                        ("flatten", nn.Flatten(1)),
                        ("out", nn.Linear(n_features[-1], num_outputs)),
                    ]
                )
            )
            if num_outputs
            else None
        )
        if self.fc:
            self.fc.output_act = get_act_layer(output_act) if output_act else None

        self._init_module(conv_type, norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.act0(x)
        x = self.pool0(x)

        for i in range(1, self.n_layers + 1):
            x = self.get_submodule(f"layer{i}")(x)

        if self.fc is not None:
            x = self.fc(x)

        return x

    def _get_block(self, block_type: Union[str, ResNetBlockType]) -> nn.Module:
        """
        Gets the residual block, depending on the block choice made by the user and depending
        on whether squeeze-excitation mode or not.
        """
        block_type = ResNetBlockType(block_type)
        if block_type == ResNetBlockType.BASIC:
            in_planes = self.n_features
            if self.squeeze_excitation:
                block = SEResNetBlock
                block.reduction = self.se_reduction
            else:
                block = ResNetBlock
        elif block_type == ResNetBlockType.BOTTLENECK:
            in_planes = bottleneck_reduce(self.n_features, self.bottleneck_reduction)
            if self.squeeze_excitation:
                block = SEResNetBottleneck
                block.reduction = self.se_reduction
            else:
                block = ResNetBottleneck
            block.expansion = self.bottleneck_reduction

        return block, in_planes  # pylint: disable=possibly-used-before-assignment

    def _get_layers(self):
        """
        Gets convolution, normalization, pooling and adaptative average pooling layers.
        """
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[
            Conv.CONV, self.spatial_dims
        ]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[
            Norm.BATCH, self.spatial_dims
        ]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[
            Pool.MAX, self.spatial_dims
        ]
        avgp_type: Type[
            Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]
        ] = Pool[Pool.ADAPTIVEAVG, self.spatial_dims]

        return conv_type, norm_type, pool_type, avgp_type

    def _make_resnet_layer(
        self,
        block: Type[Union[ResNetBlock, ResNetBottleneck]],
        planes: int,
        blocks: int,
        spatial_dims: int,
        act: ActivationParameters,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Builds a ResNet layer.
        """
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_type(  # pylint: disable=not-callable
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=self.bias_downsample,
                ),
                norm_type(planes * block.expansion),  # pylint: disable=not-callable
            )

        layers = [
            block(
                in_planes=self.in_planes,
                planes=planes,
                spatial_dims=spatial_dims,
                stride=stride,
                downsample=downsample,
                act=act,
            )
        ]

        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, planes, spatial_dims=spatial_dims, act=act)
            )

        return nn.Sequential(*layers)

    def _init_module(
        self, conv_type: Type[nn.Module], norm_type: Type[nn.Module]
    ) -> None:
        """
        Initializes the parameters.
        """
        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(
                    torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)


class ResNet(GeneralResNet):
    """
    ResNet, based on `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Adapted from :py:class:`MONAI's implementation <monai.networks.nets.ResNet>`.

    The user can customize the number of residual blocks, the number of downsampling blocks, the number of channels
    in each block, as well as other parameters like the type of residual block used.

    ResNet is a fully convolutional network that can work with an input of any size, provided that it is large
    enough not to be reduced to a 1-pixel image (before the adaptative average pooling).

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
    block_type : Union[str, ResNetBlockType], default="basic"
        Type of residual block. Either ``basic`` or ``bottleneck``. Default to ``basic``, as in ``ResNet-18``.
    n_res_blocks : Sequence[int], default=(2, 2, 2, 2)
        Number of residual block in each ResNet layer. A ResNet layer refers here to a set of residual blocks
        between two downsamplings. The length of ``n_res_blocks`` thus determines the number of ResNet layers.
        Default to ``(2, 2, 2, 2)``, as in ``ResNet-18``.
    n_features : Sequence[int], default=(64, 128, 256, 512)
        Number of output feature maps for each ResNet layer. The length of ``n_features`` must be equal to the length
        of ``n_res_blocks``. All elements of ``n_features`` must be divisible by ``bottleneck_reduction``.\n
        Default to ``(64, 128, 256, 512)``, as in ``ResNet-18``.
    init_conv_size : Union[Sequence[int], int], default=7
        Kernel size for the first convolution.
        If ``tuple``, it will be understood as the values for each dimension.
        Default to ``7``, as in the original paper.
    init_conv_stride : Union[Sequence[int], int], default=2
        Stride for the first convolution.
        If ``tuple``, it will be understood as the values for each dimension.
        Default to ``2``, as in the original paper.
    bottleneck_reduction : int, default=4
        If ``block_type="bottleneck"``, ``bottleneck_reduction`` determines the reduction factor for the number
        of feature maps in bottleneck layers (1x1 convolutions). Default to ``4``, as in the original paper.
    act : ActivationParameters, default=("relu", {"inplace": True})
        The activation function used after a convolutional layer, and optionally its arguments.
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

    Raises
    ------
    ValueError
        If ``len(n_features)!=len(n_res_blocks)``.
    ValueError
        If some elements of ``n_features`` are not divisible by ``bottleneck_reduction``.

    Examples
    --------

    .. code-block::

        >>> ResNet(
                spatial_dims=2,
                in_channels=1,
                num_outputs=2,
                block_type="bottleneck",
                bottleneck_reduction=4,
                n_features=(8, 16),
                n_res_blocks=(2, 2),
                output_act="softmax",
                init_conv_size=5,
            )
        ResNet(
            (conv0): Conv2d(1, 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
            (norm0): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act0): ReLU(inplace=True)
            (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            (layer1): Sequential(
                (0): ResNetBottleneck(
                    (conv1): Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (norm1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act1): ReLU(inplace=True)
                    (conv2): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (norm2): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act2): ReLU(inplace=True)
                    (conv3): Conv2d(2, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (norm3): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (downsample): Sequential(
                        (0): Conv2d(2, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (act3): ReLU(inplace=True)
                )
                (1): ResNetBottleneck(
                    (conv1): Conv2d(8, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (norm1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act1): ReLU(inplace=True)
                    (conv2): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (norm2): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act2): ReLU(inplace=True)
                    (conv3): Conv2d(2, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (norm3): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act3): ReLU(inplace=True)
                )
            )
            (layer2): Sequential(
                (0): ResNetBottleneck(
                    (conv1): Conv2d(8, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (norm1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act1): ReLU(inplace=True)
                    (conv2): Conv2d(4, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                    (norm2): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act2): ReLU(inplace=True)
                    (conv3): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (norm3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (downsample): Sequential(
                        (0): Conv2d(8, 16, kernel_size=(1, 1), stride=(2, 2), bias=False)
                        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (act3): ReLU(inplace=True)
                )
                (1): ResNetBottleneck(
                    (conv1): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (norm1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act1): ReLU(inplace=True)
                    (conv2): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (norm2): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act2): ReLU(inplace=True)
                    (conv3): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (norm3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (act3): ReLU(inplace=True)
                )
            )
            (fc): Sequential(
                (pool): AdaptiveAvgPool2d(output_size=(1, 1))
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (out): Linear(in_features=16, out_features=2, bias=True)
                (output_act): Softmax(dim=None)
            )
        )

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.SEResNet`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_outputs: Optional[int],
        block_type: Union[str, ResNetBlockType] = ResNetBlockType.BASIC,
        n_res_blocks: Sequence[int] = (2, 2, 2, 2),
        n_features: Sequence[int] = (64, 128, 256, 512),
        init_conv_size: Union[Sequence[int], int] = 7,
        init_conv_stride: Union[Sequence[int], int] = 2,
        bottleneck_reduction: int = 4,
        act: ActivationParameters = ("relu", {"inplace": True}),
        output_act: Optional[ActivationParameters] = None,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_outputs=num_outputs,
            block_type=block_type,
            n_res_blocks=n_res_blocks,
            n_features=n_features,
            init_conv_size=init_conv_size,
            init_conv_stride=init_conv_stride,
            bottleneck_reduction=bottleneck_reduction,
            se_reduction=None,
            act=act,
            output_act=output_act,
        )

    def _load_weights(self, url: str) -> None:
        """To load weights from torchvision."""
        fc_layers = deepcopy(self.fc)
        self.fc = None
        pretrained_dict = load_state_dict_from_url(url, progress=True)
        self.load_state_dict(_state_dict_adapter(pretrained_dict))
        self.fc = fc_layers


class ResNet18(ResNet):
    """
    ResNet-18, from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

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
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.resnet18`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ResNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=num_outputs,
            n_res_blocks=(2, 2, 2, 2),
            block_type=ResNetBlockType.BASIC,
            n_features=(64, 128, 256, 512),
            output_act=output_act,
        )
        if pretrained:
            self._load_weights(ResNet18_Weights.DEFAULT.url)


class ResNet34(ResNet):
    """
    ResNet-34, from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

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
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.resnet34`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ResNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=num_outputs,
            n_res_blocks=(3, 4, 6, 3),
            block_type=ResNetBlockType.BASIC,
            n_features=(64, 128, 256, 512),
            output_act=output_act,
        )
        if pretrained:
            self._load_weights(ResNet34_Weights.DEFAULT.url)


class ResNet50(ResNet):
    """
    ResNet-50, from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

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
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.resnet50`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ResNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
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
        if pretrained:
            self._load_weights(ResNet50_Weights.DEFAULT.url)


class ResNet101(ResNet):
    """
    ResNet-101, from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

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
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.resnet101`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ResNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
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
        if pretrained:
            self._load_weights(ResNet101_Weights.DEFAULT.url)


class ResNet152(ResNet):
    """
    ResNet-152, from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

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
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.resnet152`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ResNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
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
        if pretrained:
            self._load_weights(ResNet152_Weights.DEFAULT.url)


def bottleneck_reduce(
    n_features: Sequence[int], bottleneck_reduction: int
) -> Sequence[int]:
    """
    Finds number of feature maps for the bottleneck layers.
    """
    reduced_features = []
    for n in n_features:
        if n % bottleneck_reduction != 0:
            raise ValueError(
                "All elements of n_features must be divisible by bottleneck_reduction. "
                f"Got {n} in n_features and bottleneck_reduction={bottleneck_reduction}"
            )
        reduced_features.append(n // bottleneck_reduction)

    return reduced_features


def check_res_blocks(n_res_blocks: Sequence[int], n_features: Sequence[int]) -> None:
    """
    Checks consistency between `n_res_blocks` and `n_features`.
    """
    if not isinstance(n_res_blocks, Sequence):
        raise ValueError(f"n_res_blocks must be a sequence, got {n_res_blocks}")
    if not isinstance(n_features, Sequence):
        raise ValueError(f"n_features must be a sequence, got {n_features}")
    if len(n_features) != len(n_res_blocks):
        raise ValueError(
            f"n_features and n_res_blocks must have the same length, got n_features={n_features} "
            f"and n_res_blocks={n_res_blocks}"
        )


def _state_dict_adapter(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    A mapping between torchvision's layer names and ours.
    """
    state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}

    mappings = [
        (r"(?<!\.)conv1", "conv0"),
        (r"(?<!\.)bn1", "norm0"),
        ("bn", "norm"),
    ]

    for key in list(state_dict.keys()):
        new_key = key
        for transform in mappings:
            new_key = re.sub(transform[0], transform[1], new_key)
        state_dict[new_key] = state_dict.pop(key)

    return state_dict
