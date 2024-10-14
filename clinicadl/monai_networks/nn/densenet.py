import re
from collections import OrderedDict
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Union

import torch.nn as nn
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets import DenseNet as BaseDenseNet
from torch.hub import load_state_dict_from_url
from torchvision.models.densenet import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
)

from .utils import ActivationParameters


class DenseNet(nn.Sequential):
    """
    DenseNet based on the [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993) paper.
    Adapted from [MONAI's implementation](https://docs.monai.io/en/stable/networks.html#densenet).

    The user can customize the number of dense blocks, the number of dense layers in each block, as well as
    other parameters like the growth rate.

    DenseNet is a fully convolutional network that can work with input of any size, provided that is it large
    enough not be reduced to a 1-pixel image (before the adaptative average pooling).

    Parameters
    ----------
    spatial_dims : int
        number of spatial dimensions of the input image.
    in_channels : int
        number of channels in the input image.
    num_outputs : Optional[int]
        number of output variables after the last linear layer.\n
        If None, the features before the last fully connected layer will be returned.
    n_dense_layers : Sequence[int] (optional, default=(6, 12, 24, 16))
        number of dense layers in each dense block. Thus, this parameter also defines the number of dense blocks.
        Default is set to DenseNet-121 parameter.
    init_features : int (optional, default=64)
        number of feature maps after the initial convolution. Default is set to 64, as in the original paper.
    growth_rate : int (optional, default=32)
        how many feature maps to add at each dense layer. Default is set to 32, as in the original paper.
    bottleneck_factor : int (optional, default=4)
        multiplicative factor for bottleneck layers (1x1 convolutions). The output of of these bottleneck layers will
        have `bottleneck_factor * growth_rate` feature maps. Default is 4, as in the original paper.
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
    dropout : Optional[float] (optional, default=None)
        dropout ratio. If None, no dropout.

    Examples
    --------
    >>> DenseNet(spatial_dims=2, in_channels=1, num_outputs=2, output_act="softmax", n_dense_layers=(2, 2))
    DenseNet(
        (features): Sequential(
            (conv0): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act0): ReLU(inplace=True)
            (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            (denseblock1): _DenseBlock(
                (denselayer1): _DenseLayer(
                    (layers): Sequential(
                        (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (act1): ReLU(inplace=True)
                        (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (act2): ReLU(inplace=True)
                        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    )
                )
                (denselayer2): _DenseLayer(
                    (layers): Sequential(
                        (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (act1): ReLU(inplace=True)
                        (conv1): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (act2): ReLU(inplace=True)
                        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    )
                )
            )
            (transition1): _Transition(
                (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU(inplace=True)
                (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            (denseblock2): _DenseBlock(
                (denselayer1): _DenseLayer(
                    (layers): Sequential(
                        (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (act1): ReLU(inplace=True)
                        (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (act2): ReLU(inplace=True)
                        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    )
                )
                (denselayer2): _DenseLayer(
                    (layers): Sequential(
                        (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (act1): ReLU(inplace=True)
                        (conv1): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (act2): ReLU(inplace=True)
                        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    )
                )
            )
            (norm5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (fc): Sequential(
            (act): ReLU(inplace=True)
            (pool): AdaptiveAvgPool2d(output_size=1)
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (out): Linear(in_features=128, out_features=2, bias=True)
            (output_act): Softmax(dim=None)
        )
    )
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_outputs: Optional[int],
        n_dense_layers: Sequence[int] = (6, 12, 24, 16),
        init_features: int = 64,
        growth_rate: int = 32,
        bottleneck_factor: int = 4,
        act: ActivationParameters = ("relu", {"inplace": True}),
        output_act: ActivationParameters = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        base_densenet = BaseDenseNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_outputs if num_outputs else 1,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=n_dense_layers,
            bn_size=bottleneck_factor,
            act=act,
            dropout_prob=dropout if dropout else 0.0,
        )
        self.features = base_densenet.features
        self.fc = base_densenet.class_layers if num_outputs else None
        if self.fc:
            self.fc.output_act = get_act_layer(output_act) if output_act else None

        self._rename_act(self)

    @classmethod
    def _rename_act(cls, module: nn.Module) -> None:
        """
        Rename activation layers from 'relu' to 'act'.
        """
        for name, layer in list(module.named_children()):
            if "relu" in name:
                module._modules = OrderedDict(
                    [
                        (key.replace("relu", "act"), sub_m)
                        for key, sub_m in module._modules.items()
                    ]
                )
            else:
                cls._rename_act(layer)


class CommonDenseNet(str, Enum):
    """Supported DenseNet networks."""

    DENSENET_121 = "DenseNet-121"
    DENSENET_161 = "DenseNet-161"
    DENSENET_169 = "DenseNet-169"
    DENSENET_201 = "DenseNet-201"


def get_densenet(
    model: Union[str, CommonDenseNet],
    num_outputs: Optional[int],
    output_act: ActivationParameters = None,
    pretrained: bool = False,
) -> DenseNet:
    """
    To get a DenseNet implemented in the [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993)
    paper.

    Only the last fully connected layer will be changed to match `num_outputs`.

    The user can also use the pretrained models from `torchvision`. Note that the last fully connected layer will not
    used pretrained weights, as it is task specific.

    .. warning:: `DenseNet-121`, `DenseNet-161`, `DenseNet-169` and `DenseNet-201` only works with 2D images with 3 channels.

    Notes: `torchvision` does not provide an implementation for `DenseNet-264` but provides a `DenseNet-161` that is not
    mentioned in the paper.

    Parameters
    ----------
    model : Union[str, CommonDenseNet]
        The name of the DenseNet. Available networks are `DenseNet-121`, `DenseNet-161`, `DenseNet-169` and `DenseNet-201`.
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
        pytorch.org/vision/main/models/densenet.html).

    Returns
    -------
    DenseNet
        The network, with potentially pretrained weights.
    """
    model = CommonDenseNet(model)
    if model == CommonDenseNet.DENSENET_121:
        n_dense_layers = (6, 12, 24, 16)
        growth_rate = 32
        init_features = 64
        model_url = DenseNet121_Weights.DEFAULT.url
    elif model == CommonDenseNet.DENSENET_161:
        n_dense_layers = (6, 12, 36, 24)
        growth_rate = 48
        init_features = 96
        model_url = DenseNet161_Weights.DEFAULT.url
    elif model == CommonDenseNet.DENSENET_169:
        n_dense_layers = (6, 12, 32, 32)
        growth_rate = 32
        init_features = 64
        model_url = DenseNet169_Weights.DEFAULT.url
    elif model == CommonDenseNet.DENSENET_201:
        n_dense_layers = (6, 12, 48, 32)
        growth_rate = 32
        init_features = 64
        model_url = DenseNet201_Weights.DEFAULT.url

    densenet = DenseNet(
        spatial_dims=2,
        in_channels=3,
        num_outputs=num_outputs,
        n_dense_layers=n_dense_layers,
        growth_rate=growth_rate,
        init_features=init_features,
        output_act=output_act,
    )
    if not pretrained:
        return densenet

    pretrained_dict = load_state_dict_from_url(model_url, progress=True)
    features_state_dict = {
        k.replace("features.", ""): v
        for k, v in pretrained_dict.items()
        if "classifier" not in k
    }
    densenet.features.load_state_dict(_state_dict_adapter(features_state_dict))

    return densenet


def _state_dict_adapter(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    To update the old nomenclature in the pretrained state dict.
    Adapted from `_load_state_dict` in [torchvision.models.densenet](https://pytorch.org/vision/main
    /_modules/torchvision/models/densenet.html).
    """
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            new_key = re.sub(r"^(.*denselayer\d+)\.", r"\1.layers.", new_key)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    return state_dict
