import re
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence

import torch.nn as nn
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets import DenseNet as BaseDenseNet
from pydantic import NonNegativeFloat, PositiveInt
from torch.hub import load_state_dict_from_url
from torchvision.models.densenet import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
)

from clinicadl.utils.factories import get_defaults_from

from .layers.utils import ActivationParameters
from .utils.config import (
    NetworkConfig,
    _DropoutConfig,
    _SpatialDimsConfig,
)

__all__ = ["DenseNet", "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201"]


class DenseNet(nn.Sequential):
    """
    DenseNet, based on `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Adapted from :py:class:`MONAI's implementation <monai.networks.nets.DenseNet>`.

    The user can customize the number of dense blocks, the number of dense layers in each block, as well as
    other parameters like the growth rate.

    DenseNet is a fully convolutional network that can work with an input of any size, provided that it is large
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
    n_dense_layers : Sequence[int], default=(6, 12, 24, 16)
        Number of dense layers in each dense block. Thus, this parameter also defines the number of dense blocks
        (equal to the length of the sequence). Default is set to the value of ``DenseNet-121``.
    init_features : int, default=64
        Number of feature maps after the initial convolution. Default is set to ``64``, as in the original paper.
    growth_rate : int, default=32
        How many feature maps to add at each dense layer. Default is set to ``32``, as in the original paper.
    bottleneck_factor : int, default=4
        Multiplicative factor for bottleneck layers (1x1 convolutions). The output of of these bottleneck layers will
        have ``bottleneck_factor * growth_rate`` feature maps. Default is ``4``, as in the original paper.
    act : ActivationParameters, default=("relu", {"inplace": True})
        The activation function used after a convolutional layer, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.\n
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

    .. code-block::

        >>> DenseNet(
                spatial_dims=2,
                in_channels=1,
                num_outputs=2,
                output_act="softmax",
                n_dense_layers=(2, 2),
            )
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
        output_act: Optional[ActivationParameters] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.config = DenseNetConfig(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_outputs=num_outputs,
            n_dense_layers=n_dense_layers,
            init_features=init_features,
            growth_rate=growth_rate,
            bottleneck_factor=bottleneck_factor,
            act=act,
            output_act=output_act,
            dropout=dropout,
        )

        base_densenet = BaseDenseNet(
            spatial_dims=self.config.spatial_dims,
            in_channels=self.config.in_channels,
            out_channels=self.config.num_outputs or 1,
            init_features=self.config.init_features,
            growth_rate=self.config.growth_rate,
            block_config=self.config.n_dense_layers,
            bn_size=self.config.bottleneck_factor,
            act=self.config.act,
            dropout_prob=self.config.dropout or 0.0,
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
                module._modules = OrderedDict(  # pylint: disable=protected-access
                    [
                        (key.replace("relu", "act"), sub_m)
                        for key, sub_m in module._modules.items()  # pylint: disable=protected-access
                    ]
                )
            else:
                cls._rename_act(layer)

    def _load_weights(self, url: str) -> None:
        """To load weights from torchvision."""
        pretrained_dict = load_state_dict_from_url(url, progress=True)
        features_state_dict = {
            k.replace("features.", ""): v
            for k, v in pretrained_dict.items()
            if "classifier" not in k
        }
        self.features.load_state_dict(_state_dict_adapter(features_state_dict))


class DenseNet121(DenseNet):
    """
    DenseNet-121, from `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

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
        from :py:func:`torchvision.models.densenet121`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.DenseNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        config = DenseNet121Config(
            num_outputs=num_outputs, output_act=output_act, pretrained=pretrained
        )
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=config.num_outputs,
            n_dense_layers=(6, 12, 24, 16),
            growth_rate=32,
            init_features=64,
            output_act=config.output_act,
        )
        if config.pretrained:
            self._load_weights(DenseNet121_Weights.DEFAULT.url)


class DenseNet161(DenseNet):
    """
    DenseNet-161, from `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

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
        from :py:func:`torchvision.models.densenet161`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.DenseNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        config = DenseNet161Config(
            num_outputs=num_outputs, output_act=output_act, pretrained=pretrained
        )
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=config.num_outputs,
            n_dense_layers=(6, 12, 36, 24),
            growth_rate=48,
            init_features=96,
            output_act=config.output_act,
        )
        if config.pretrained:
            self._load_weights(DenseNet161_Weights.DEFAULT.url)


class DenseNet169(DenseNet):
    """
    DenseNet-161, from `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

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
        from :py:func:`torchvision.models.densenet169`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.DenseNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        config = DenseNet169Config(
            num_outputs=num_outputs, output_act=output_act, pretrained=pretrained
        )
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=config.num_outputs,
            n_dense_layers=(6, 12, 32, 32),
            growth_rate=32,
            init_features=64,
            output_act=config.output_act,
        )
        if config.pretrained:
            self._load_weights(DenseNet169_Weights.DEFAULT.url)


class DenseNet201(DenseNet):
    """
    DenseNet-201, from `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

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
        from :py:func:`torchvision.models.densenet201`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.DenseNet`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        config = DenseNet201Config(
            num_outputs=num_outputs, output_act=output_act, pretrained=pretrained
        )
        super().__init__(
            spatial_dims=2,
            in_channels=3,
            num_outputs=config.num_outputs,
            n_dense_layers=(6, 12, 48, 32),
            growth_rate=32,
            init_features=64,
            output_act=config.output_act,
        )
        if config.pretrained:
            self._load_weights(DenseNet201_Weights.DEFAULT.url)


DENSE_NET_DEFAULTS = get_defaults_from(DenseNet)
DENSE_NET_121_DEFAULTS = get_defaults_from(DenseNet121)
DENSE_NET_161_DEFAULTS = get_defaults_from(DenseNet161)
DENSE_NET_169_DEFAULTS = get_defaults_from(DenseNet169)
DENSE_NET_201_DEFAULTS = get_defaults_from(DenseNet201)


class DenseNetConfig(
    NetworkConfig,
    _SpatialDimsConfig,
    _DropoutConfig,
):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt
    num_outputs: Optional[PositiveInt]
    n_dense_layers: Sequence[PositiveInt] = DENSE_NET_DEFAULTS["n_dense_layers"]
    init_features: PositiveInt = DENSE_NET_DEFAULTS["init_features"]
    growth_rate: PositiveInt = DENSE_NET_DEFAULTS["growth_rate"]
    bottleneck_factor: PositiveInt = DENSE_NET_DEFAULTS["bottleneck_factor"]
    act: ActivationParameters = DENSE_NET_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = DENSE_NET_DEFAULTS["output_act"]
    dropout: Optional[NonNegativeFloat] = DENSE_NET_DEFAULTS["dropout"]

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return DenseNet


class DenseNet121Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet121`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = DENSE_NET_121_DEFAULTS["output_act"]
    pretrained: bool = DENSE_NET_121_DEFAULTS["pretrained"]

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return DenseNet121


class DenseNet161Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet161`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = DENSE_NET_161_DEFAULTS["output_act"]
    pretrained: bool = DENSE_NET_161_DEFAULTS["pretrained"]

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return DenseNet161


class DenseNet169Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet169`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = DENSE_NET_169_DEFAULTS["output_act"]
    pretrained: bool = DENSE_NET_169_DEFAULTS["pretrained"]

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return DenseNet169


class DenseNet201Config(NetworkConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.DenseNet201`.
    """

    num_outputs: Optional[PositiveInt]
    output_act: Optional[ActivationParameters] = DENSE_NET_201_DEFAULTS["output_act"]
    pretrained: bool = DENSE_NET_201_DEFAULTS["pretrained"]

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return DenseNet201


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
