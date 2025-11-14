from typing import Callable, Optional, Sequence

import numpy as np
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_act_layer, get_pool_layer
from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from clinicadl.networks.nn.layers.utils import (
    ActivationParameters,
    ConvNormalizationParameters,
)
from clinicadl.utils.factories import get_defaults_from

from .layers.utils import (
    ActFunction,
    ActivationParameters,
    ConvNormalizationParameters,
    ConvNormLayer,
    ConvParameters,
    PoolingLayer,
    PoolingParameters,
    SingleLayerPoolingParameters,
)
from .utils import (
    calculate_conv_out_shape,
    calculate_pool_out_shape,
    check_adn_ordering,
    check_norm_layer,
    check_pool_indices,
    ensure_list_of_tuples,
)
from .utils.config import NetworkConfig, _DropoutConfig, _SpatialDimsConfig


class ConvEncoder(nn.Sequential):
    """
    Fully convolutional encoder network with convolutional, pooling, normalization, activation
    and dropout layers.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of channels in the input image.
    channels : Sequence[int]
        Number of output channels of each convolutional layer. Thus, this
        parameter also controls the number of convolutional layers (equal to the length of the sequence).
    kernel_size : ConvParameters, default=3
        Kernel size of the convolutional layers. Can be an ``int``, a ``tuple``, or a ``list``:

        - ``int``: the value will be used for all layers and all dimensions;
        - ``tuple`` (e.g. ``(3, 3, 2)``): it will be interpreted as the values for each dimension. These values
          will be used for all the layers;
        - ``list`` (e.g. ``[(3, 3, 2), 3]``): it will be interpreted as the kernel sizes for each layer.
          The length of the list must be equal to the number of convolutional layers (i.e. ``len(channels)``).
    stride : ConvParameters, default=1
        Stride of the convolutional layers. Can be an ``int``, a ``tuple``, or a ``list``, and is passed in the same way
        as ``kernel_size``.\n
    padding : ConvParameters, default=0
        Padding of the convolutional layers. Can be an ``int``, a ``tuple``, or a ``list``, and is passed in the same way
        as ``kernel_size``.\n
    dilation : ConvParameters, default=1
        Dilation factor of the convolutional layers. Can be an ``int``, a ``tuple``, or a ``list``, and is passed in the same way
        as ``kernel_size``.\n
    pooling : Optional[PoolingParameters], default=("max", {"kernel_size": 2})
        The pooling mode and the arguments of the pooling layer, passed as ``(pooling_mode, arguments)``,  where ``arguments`` is a dictionary.
        If ``None``, no pooling will be performed in the network.\n
        ``pooling_mode`` can be any value in {``max``, ``avg``, ``adaptivemax``, ``adaptiveavg``}. Please refer to
        :torch:`PyTorch pooling layers <nn.html#pooling-layers>` to know the arguments for each of them.\n
        If a ``list`` is passed, it will be understood as the pooling for each pooling layer.
    pooling_indices : Optional[Sequence[int]], default=None
        Indices of the convolutional layers after which pooling should be performed.
        If ``None``, no pooling will be performed. An index equal to ``-1`` will be understood as a pooling layer before
        the first convolution.
    act : Optional[ActivationParameters], default="prelu"
        The activation function used after a convolutional layer, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network. Must be passed in the same way as ``act``.
        If ``None``, no last activation will be applied.
    norm : Optional[ConvNormalizationParameters], default="instance"
        The normalization layer used after a convolutional layer, and optionally its arguments.
        Must be passed as ``norm_type`` or ``(norm_type, arguments)`` where ``arguments`` is a dictionary.
        If ``None``, no normalization will be performed.\n
        ``norm_type`` can be any value in {``batch``, ``group``, ``instance``, ``syncbatch``}. Please refer to
        :torch:`PyTorch normalization layers <nn.html#normalization-layers>` to know the arguments for each of them.

        .. note::
            Please note that there's no need to pass the arguments ``num_channels`` and ``num_features``
            of the normalization layer, as they are automatically inferred from the output of the previous layer in the network.

    dropout : Optional[float], default=None
        Dropout ratio. If ``None``, no dropout.
    bias : bool, default=True
        Whether to have a bias term in linear layers.
    adn_ordering : str, default="NDA"
        Order of operations Activation, Dropout and Normalization, after a linear layer (except the last
        one).  **Cannot contain duplicated letters**.
        For example if ``"ND"`` is passed, Normalization and then Dropout will be performed (without Activation).\n

        .. note::
            ADN will not be applied after the last linear layer.

    Raises
    ------
    ValueError
        If a ``list`` is passed for ``kernel_size``, ``stride``, ``padding``, or ``dilation``, and the size of this
        list in not equal to ``len(channels)``.
    ValueError
        If indices in ``pooling_indices`` are greater than ``len(channels)-1`` (``len(channels)-1`` being the index of the last
        convolution layer).
    ValueError
        If a ``list`` is passed for ``pooling``, and ``len(pooling)!=len(pooling_indices)``.
    ValueError
        If the activation or normalization layer requires a mandatory argument, which is not passed by the user (via a dictionary
        in ``act`` or ``norm``).

    Examples
    --------

    .. code-block:: python

        >>> ConvEncoder(
                spatial_dims=2,
                in_channels=1,
                channels=[2, 4, 8],
                kernel_size=(3, 5),
                stride=1,
                padding=[1, (0, 1), 0],
                dilation=1,
                pooling=[("max", {"kernel_size": 2}), ("avg", {"kernel_size": 2})],
                pooling_indices=[0, 1],
                act="elu",
                output_act="relu",
                norm=("batch", {"eps": 1e-05}),
                dropout=0.1,
                bias=True,
                adn_ordering="NDA",
            )
        ConvEncoder(
            (layer0): Convolution(
                (conv): Conv2d(1, 2, kernel_size=(3, 5), stride=(1, 1), padding=(1, 1))
                (adn): ADN(
                    (N): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (D): Dropout(p=0.1, inplace=False)
                    (A): ELU(alpha=1.0)
                )
            )
            (pool0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (layer1): Convolution(
                (conv): Conv2d(2, 4, kernel_size=(3, 5), stride=(1, 1), padding=(0, 1))
                (adn): ADN(
                    (N): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (D): Dropout(p=0.1, inplace=False)
                    (A): ELU(alpha=1.0)
                )
            )
            (pool1): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (layer2): Convolution(
                (conv): Conv2d(4, 8, kernel_size=(3, 5), stride=(1, 1))
            )
            (output_act): ReLU()
        )

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ConvDecoder`
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: ConvParameters = 3,
        stride: ConvParameters = 1,
        padding: ConvParameters = 0,
        dilation: ConvParameters = 1,
        pooling: Optional[PoolingParameters] = (
            PoolingLayer.MAX,
            {"kernel_size": 2},
        ),
        pooling_indices: Optional[Sequence[int]] = None,
        act: Optional[ActivationParameters] = ActFunction.PRELU,
        output_act: Optional[ActivationParameters] = None,
        norm: Optional[ConvNormalizationParameters] = ConvNormLayer.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
        adn_ordering: str = "NDA",
        _input_size: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        self.config = ConvEncoderConfig(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            pooling=pooling,
            pooling_indices=pooling_indices,
            act=act,
            output_act=output_act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

        self._current_size = _input_size
        self._size_details = [self._current_size] if _input_size else None

        n_poolings = 0
        if self.config.pooling and -1 in self.config.pooling_indices:
            pooling_layer = self._get_pool_layer(self.config.pooling[n_poolings])
            self.add_module("init_pool", pooling_layer)
            n_poolings += 1

        echannel = self.config.in_channels
        for i, (c, k, s, p, d) in enumerate(
            zip(
                self.config.channels,
                self.config.kernel_size,
                self.config.stride,
                self.config.padding,
                self.config.dilation,
            )
        ):
            conv_layer = self._get_conv_layer(
                in_channels=echannel,
                out_channels=c,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                is_last=(i == len(channels) - 1),
            )
            self.add_module(f"layer{i}", conv_layer)
            echannel = c  # use the output channel number as the input for the next loop
            if self.config.pooling and i in self.config.pooling_indices:
                pooling_layer = self._get_pool_layer(self.config.pooling[n_poolings])
                self.add_module(f"pool{i}", pooling_layer)
                n_poolings += 1

        self.output_act = get_act_layer(output_act) if output_act else None

    @property
    def _final_size(self):
        """
        To know the size of an image at the end of the network.
        """
        return self._current_size

    @_final_size.setter
    def _final_size(self, fct: Callable[[tuple[int, ...]], tuple[int, ...]]):
        """
        Takes as input the function used to update the current image size.
        """
        if self._current_size is not None:
            self._current_size = fct(self._current_size)
            self._size_details.append(self._current_size)
        self._check_size()

    def _get_conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        is_last: bool,
    ) -> Convolution:
        """
        Gets the parametrized Convolution-ADN block and updates the current output size.
        """
        self._final_size = lambda size: calculate_conv_out_shape(
            size, kernel_size, stride, padding, dilation
        )

        return Convolution(
            conv_only=is_last,
            spatial_dims=self.config.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            act=self.config.act,
            norm=self.config.norm,
            dropout=self.config.dropout,
            bias=self.config.bias,
            adn_ordering=self.config.adn_ordering,
        )

    def _get_pool_layer(self, pooling: SingleLayerPoolingParameters) -> nn.Module:
        """
        Gets the parametrized pooling layer and updates the current output size.
        """
        pool_layer = get_pool_layer(pooling, spatial_dims=self.config.spatial_dims)
        old_size = self._final_size
        self._final_size = lambda size: calculate_pool_out_shape(
            pool_mode=pooling[0], in_shape=size, **pool_layer.__dict__
        )

        if (
            self._final_size is not None
            and (np.array(old_size) < np.array(self._final_size)).any()
        ):
            raise ValueError(
                f"You passed {pooling} as a pooling layer. But before this layer, the size of the image "
                f"was {old_size}. So, pooling can't be performed."
            )

        return pool_layer

    def _check_size(self) -> None:
        """
        Checks that image size never reaches 0.
        """
        if self._current_size is not None and (np.array(self._current_size) <= 0).any():
            raise ValueError(
                f"Failed to build the network. An image of size 0 or less has been reached. Stopped at:\n {self}"
            )


CONV_ENCODER_DEFAULTS = get_defaults_from(ConvEncoder)


class _BaseConvOptions(_DropoutConfig):
    """
    Base config class for ConvEncoder and ConvDecoder options.
    """

    channels: Sequence[PositiveInt]
    kernel_size: ConvParameters
    stride: ConvParameters
    padding: ConvParameters
    dilation: ConvParameters
    norm: Optional[ConvNormalizationParameters]
    adn_ordering: str

    @field_validator("norm", mode="after")
    @classmethod
    def _norm_validator(cls, v):
        return check_norm_layer(v)

    @field_validator("adn_ordering")
    @classmethod
    def _adn_ordering_validator(cls, v):
        return check_adn_ordering(v)

    def _check_args_dim(self, dim: int) -> None:
        n_layers = len(self.channels)
        self.__dict__["kernel_size"] = ensure_list_of_tuples(
            self.kernel_size, dim, n_layers, "kernel_size"
        )
        self.__dict__["stride"] = ensure_list_of_tuples(
            self.stride, dim, n_layers, "stride"
        )
        self.__dict__["padding"] = ensure_list_of_tuples(
            self.padding, dim, n_layers, "padding"
        )
        self.__dict__["dilation"] = ensure_list_of_tuples(
            self.dilation, dim, n_layers, "dilation"
        )

    def _check_pool_indices(self, indices: Optional[Sequence[int]]) -> Sequence[int]:
        return check_pool_indices(indices, n_layers=len(self.channels))


class ConvEncoderOptions(_BaseConvOptions):
    """
    Config class for ConvEncoder when it is a submodule.
    See for example: :py:class:`clinicadl.networks.nn.CNN`
    """

    channels: Sequence[PositiveInt]
    kernel_size: ConvParameters = CONV_ENCODER_DEFAULTS["kernel_size"]
    stride: ConvParameters = CONV_ENCODER_DEFAULTS["stride"]
    padding: ConvParameters = CONV_ENCODER_DEFAULTS["padding"]
    dilation: ConvParameters = CONV_ENCODER_DEFAULTS["dilation"]
    pooling: Optional[PoolingParameters] = CONV_ENCODER_DEFAULTS["pooling"]
    pooling_indices: Sequence[int] = CONV_ENCODER_DEFAULTS["pooling_indices"]
    act: Optional[ActivationParameters] = CONV_ENCODER_DEFAULTS["act"]
    output_act: Optional[ActivationParameters] = CONV_ENCODER_DEFAULTS["output_act"]
    norm: Optional[ConvNormalizationParameters] = CONV_ENCODER_DEFAULTS["norm"]
    dropout: Optional[NonNegativeFloat] = CONV_ENCODER_DEFAULTS["dropout"]
    bias: bool = CONV_ENCODER_DEFAULTS["bias"]
    adn_ordering: str = CONV_ENCODER_DEFAULTS["adn_ordering"]

    @field_validator("pooling_indices", mode="before")
    @classmethod
    def _handle_none_pooling_indices(cls, v):
        return v or []

    @model_validator(mode="after")
    def check_pooling(self):
        checked_indices = self._check_pool_indices(self.pooling_indices)
        self.__dict__["pooling"] = self._check_pool_layers(
            self.pooling, pooling_indices=checked_indices
        )

        return self

    @classmethod
    def _check_pool_layers(
        cls, pooling: PoolingParameters, pooling_indices: Sequence[int]
    ) -> list[SingleLayerPoolingParameters]:
        """
        Checks pooling arguments.
        """
        if isinstance(pooling, list):
            for pool_layer in pooling:
                cls._check_single_pool_layer(pool_layer)
            if len(pooling) != len(pooling_indices):
                raise ValueError(
                    "If you pass a list for pooling, the size of that list must match "
                    f"the size of pooling_indices. Got: pooling={pooling} and "
                    f"pooling_indices={pooling_indices}"
                )
        elif isinstance(pooling, tuple):
            cls._check_single_pool_layer(pooling)
            pooling = [pooling] * len(pooling_indices)

        return pooling

    @staticmethod
    def _check_single_pool_layer(pooling: SingleLayerPoolingParameters) -> None:
        """
        Checks pooling arguments for a single pooling layer.
        """
        pooling_type = pooling[0]
        args = pooling[1]
        if (
            pooling_type == PoolingLayer.MAX or pooling_type == PoolingLayer.AVG
        ) and "kernel_size" not in args:
            raise ValueError(
                f"For {pooling_type} pooling mode, `kernel_size` argument must be passed. "
                f"Got {args}"
            )
        elif (
            pooling_type == PoolingLayer.ADAPT_AVG
            or pooling_type == PoolingLayer.ADAPT_MAX
        ) and "output_size" not in args:
            raise ValueError(
                f"For {pooling_type} pooling mode, `output_size` argument must be passed. "
                f"Got {args}"
            )


class ConvEncoderConfig(NetworkConfig, ConvEncoderOptions, _SpatialDimsConfig):
    """
    Config class for :py:class:`clinicadl.networks.nn.ConvEncoder`.
    """

    spatial_dims: PositiveInt
    in_channels: PositiveInt

    @model_validator(mode="after")
    def _check_dim(self):
        self._check_args_dim(self.spatial_dims)
        return self

    @classmethod
    def _get_class(cls) -> type[nn.Module]:
        """Returns the network associated to this config class."""
        return ConvEncoder
