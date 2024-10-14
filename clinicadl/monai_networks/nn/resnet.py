import re
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_act_layer
from monai.utils import ensure_tuple_rep
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from .layers.resnet import ResNetBlock, ResNetBottleneck
from .layers.senet import SEResNetBlock, SEResNetBottleneck
from .layers.utils import ActivationParameters


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

        self._check_args_consistency(n_res_blocks, n_features)
        self.squeeze_excitation = True if se_reduction else False
        self.se_reduction = se_reduction
        self.n_features = n_features
        self.bottleneck_reduction = bottleneck_reduction
        self.spatial_dims = spatial_dims

        block, in_planes = self._get_block(block_type)

        conv_type, norm_type, pool_type, avgp_type = self._get_layers()

        block_avgpool = [0, 1, (1, 1), (1, 1, 1)]

        self.in_planes = in_planes[0]
        self.n_layers = len(in_planes)
        self.bias_downsample = False

        conv1_kernel_size = ensure_tuple_rep(init_conv_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(init_conv_stride, spatial_dims)

        self.conv0 = conv_type(  # pylint: disable=not-callable
            in_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=tuple(k // 2 for k in conv1_kernel_size),
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
            in_planes = self._bottleneck_reduce(
                self.n_features, self.bottleneck_reduction
            )
            if self.squeeze_excitation:
                block = SEResNetBottleneck
                block.reduction = self.se_reduction
            else:
                block = ResNetBottleneck
            block.expansion = self.bottleneck_reduction

        return block, in_planes

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

    @classmethod
    def _bottleneck_reduce(
        cls, n_features: Sequence[int], bottleneck_reduction: int
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

    @classmethod
    def _check_args_consistency(
        cls, n_res_blocks: Sequence[int], n_features: Sequence[int]
    ) -> None:
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


class ResNet(GeneralResNet):
    """
    ResNet based on the [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) paper.
    Adapted from [MONAI's implementation](https://docs.monai.io/en/stable/networks.html#resnet).

    The user can customize the number of residual blocks, the number of downsampling blocks, the number of channels
    in each block, as well as other parameters like the type of residual block used.

    ResNet is a fully convolutional network that can work with input of any size, provided that is it large
    enough not be reduced to a 1-pixel image (before the adaptative average pooling).

    Parameters
    ----------
    spatial_dims : int
        number of spatial dimensions of the input image.
    in_channels : int
        number of channels in the input image.
    num_outputs : Optional[int]
        number of output variables after the last linear layer.\n
        If None, the features before the last fully connected layer (including average pooling) will be returned.
    block_type : Union[str, ResNetBlockType] (optional, default=ResNetBlockType.BASIC)
        type of residual block. Either `basic` or `bottleneck`. Default to `basic`, as in `ResNet-18`.
    n_res_blocks : Sequence[int] (optional, default=(2, 2, 2, 2))
        number of residual block in each ResNet layer. A ResNet layer refers here to the set of residual blocks
        between two downsamplings. The length of `n_res_blocks` thus determines the number of ResNet layers.
        Default to `(2, 2, 2, 2)`, as in `ResNet-18`.
    n_features : Sequence[int] (optional, default=(64, 128, 256, 512))
        number of output feature maps for each ResNet layer. The length of `n_features` must be equal to the length
        of `n_res_blocks`. Default to `(64, 128, 256, 512)`, as in `ResNet-18`.
    init_conv_size : Union[Sequence[int], int] (optional, default=7)
        kernel_size for the first convolution.
        If tuple, it will be understood as the values for each dimension.
        Default to 7, as in the original paper.
    init_conv_stride : Union[Sequence[int], int] (optional, default=2)
        stride for the first convolution.
        If tuple, it will be understood as the values for each dimension.
        Default to 2, as in the original paper.
    bottleneck_reduction : int (optional, default=4)
        if `block_type='bottleneck'`, `bottleneck_reduction` determines the reduction factor for the number
        of feature maps in bottleneck layers (1x1 convolutions). Default to 4, as in the original paper.
    act : ActivationParameters (optional, default=("relu", {"inplace": True}))
        the activation function used in the convolutional part, and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`.
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.\n
        Default is "relu", as in the original paper.
    output_act : ActivationParameters (optional, default=None)
        if `num_outputs` is not None, a potential activation layer applied to the outputs of the network.
        Should be pass in the same way as `act`.
        If None, no last activation will be applied.

    Examples
    --------
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
        output_act: ActivationParameters = None,
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


class CommonResNet(str, Enum):
    """Supported ResNet networks."""

    RESNET_18 = "ResNet-18"
    RESNET_34 = "ResNet-34"
    RESNET_50 = "ResNet-50"
    RESNET_101 = "ResNet-101"
    RESNET_152 = "ResNet-152"


def get_resnet(
    model: Union[str, CommonResNet],
    num_outputs: Optional[int],
    output_act: ActivationParameters = None,
    pretrained: bool = False,
) -> ResNet:
    """
    To get a ResNet implemented in the [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)
    paper.

    Only the last fully connected layer will be changed to match `num_outputs`.

    The user can also use the pretrained models from `torchvision`. Note that the last fully connected layer will not
    used pretrained weights, as it is task specific.

    .. warning:: `ResNet-18`, `ResNet-34`, `ResNet-50`, `ResNet-101` and `ResNet-152` only works with 2D images with 3
    channels.

    Parameters
    ----------
    model : Union[str, CommonResNet]
        The name of the ResNet. Available networks are `ResNet-18`, `ResNet-34`, `ResNet-50`, `ResNet-101` and `ResNet-152`.
    num_outputs : Optional[int]
        number of output variables after the last linear layer.\n
        If None, the features before the last fully connected layer will be returned.
    output_act : ActivationParameters (optional, default=None)
        if `num_outputs` is not None, a potential activation layer applied to the outputs of the network,
        and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.
    pretrained : bool (optional, default=False)
        whether to use pretrained weights. The pretrained weights used are the default ones from [torchvision](https://
        pytorch.org/vision/main/models/resnet.html).

    Returns
    -------
    ResNet
        The network, with potentially pretrained weights.
    """
    model = CommonResNet(model)
    if model == CommonResNet.RESNET_18:
        block_type = ResNetBlockType.BASIC
        n_res_blocks = (2, 2, 2, 2)
        n_features = (64, 128, 256, 512)
        model_url = ResNet18_Weights.DEFAULT.url
    elif model == CommonResNet.RESNET_34:
        block_type = ResNetBlockType.BASIC
        n_res_blocks = (3, 4, 6, 3)
        n_features = (64, 128, 256, 512)
        model_url = ResNet34_Weights.DEFAULT.url
    elif model == CommonResNet.RESNET_50:
        block_type = ResNetBlockType.BOTTLENECK
        n_res_blocks = (3, 4, 6, 3)
        n_features = (256, 512, 1024, 2048)
        model_url = ResNet50_Weights.DEFAULT.url
    elif model == CommonResNet.RESNET_101:
        block_type = ResNetBlockType.BOTTLENECK
        n_res_blocks = (3, 4, 23, 3)
        n_features = (256, 512, 1024, 2048)
        model_url = ResNet101_Weights.DEFAULT.url
    elif model == CommonResNet.RESNET_152:
        block_type = ResNetBlockType.BOTTLENECK
        n_res_blocks = (3, 8, 36, 3)
        n_features = (256, 512, 1024, 2048)
        model_url = ResNet152_Weights.DEFAULT.url

    resnet = ResNet(
        spatial_dims=2,
        in_channels=3,
        num_outputs=num_outputs,
        n_res_blocks=n_res_blocks,
        block_type=block_type,
        n_features=n_features,
        output_act=output_act,
    )
    if pretrained:
        fc_layers = deepcopy(resnet.fc)
        resnet.fc = None
        pretrained_dict = load_state_dict_from_url(model_url, progress=True)
        resnet.load_state_dict(_state_dict_adapter(pretrained_dict))
        resnet.fc = fc_layers

    return resnet


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
