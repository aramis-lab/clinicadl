from collections.abc import Sequence
from typing import Callable, Optional, Tuple

import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_act_layer
from monai.utils.misc import ensure_tuple

from .layers.unpool import get_unpool_layer
from .layers.utils import (
    ActFunction,
    ActivationParameters,
    ConvNormalizationParameters,
    ConvNormLayer,
    ConvParameters,
    NormLayer,
    SingleLayerUnpoolingParameters,
    UnpoolingLayer,
    UnpoolingParameters,
)
from .utils import (
    calculate_convtranspose_out_shape,
    calculate_unpool_out_shape,
    check_norm_layer,
    check_pool_indices,
    ensure_list_of_tuples,
)


class ConvDecoder(nn.Sequential):
    """
    Fully convolutional decoder network with transposed convolutions, unpooling, normalization, activation
    and dropout layers.

    Parameters
    ----------
    spatial_dims : int
        number of spatial dimensions of the input image.
    in_channels : int
        number of channels in the input image.
    channels : Sequence[int]
        sequence of integers stating the output channels of each transposed convolution. Thus, this
        parameter also controls the number of transposed convolutions.
    kernel_size : ConvParameters (optional, default=3)
        the kernel size of the transposed convolutions. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the kernel sizes for each layer.
        The length of the list must be equal to the number of transposed convolution layers (i.e.
        `len(channels)`).
    stride : ConvParameters (optional, default=1)
        the stride of the transposed convolutions. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the strides for each layer.
        The length of the list must be equal to the number of transposed convolution layers (i.e.
        `len(channels)`).
    padding : ConvParameters (optional, default=0)
        the padding of the transposed convolutions. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the paddings for each layer.
        The length of the list must be equal to the number of transposed convolution layers (i.e.
        `len(channels)`).
    output_padding : ConvParameters (optional, default=0)
        the output padding of the transposed convolutions. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the output paddings for each layer.
        The length of the list must be equal to the number of transposed convolution layers (i.e.
        `len(channels)`).
    dilation : ConvParameters (optional, default=1)
        the dilation factor of the transposed convolutions. Can be an integer, a tuple or a list.\n
        If integer, the value will be used for all layers and all dimensions.\n
        If tuple (of integers), it will be interpreted as the values for each dimension. These values
        will be used for all the layers.\n
        If list (of tuples or integers), it will be interpreted as the dilations for each layer.
        The length of the list must be equal to the number of transposed convolution layers (i.e.
        `len(channels)`).
    unpooling : Optional[UnpoolingParameters] (optional, default=(UnpoolingLayer.UPSAMPLE, {"scale_factor": 2}))
        the unpooling mode and the arguments of the unpooling layer, passed as `(unpooling_mode, arguments)`.
        If None, no unpooling will be performed in the network.\n
        `unpooling_mode` can be either `upsample` or `convtranspose`. Please refer to PyTorch's [Upsample]
        (https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html) or [ConvTranspose](https://
        pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) to know the mandatory and optional
        arguments.\n
        If a list is passed, it will be understood as `(unpooling_mode, arguments)` for each unpooling layer.\n
        Note: no need to pass `in_channels` and `out_channels` for `convtranspose` because the unpooling
        layers are not intended to modify the number of channels.
    unpooling_indices : Optional[Sequence[int]] (optional, default=None)
        indices of the transposed convolution layers after which unpooling should be performed.
        If None, no unpooling will be performed. An index equal to -1 will be understood as a pooling layer before
        the first transposed convolution.
    act : Optional[ActivationParameters] (optional, default=ActFunction.PRELU)
        the activation function used after a transposed convolution layer, and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.
    output_act : Optional[ActivationParameters] (optional, default=None)
        a potential activation layer applied to the output of the network. Should be pass in the same way as `act`.
        If None, no last activation will be applied.
    norm : Optional[ConvNormalizationParameters] (optional, default=NormLayer.INSTANCE)
        the normalization type used after a transposed convolution layer, and optionally the arguments of the normalization
        layer. Should be passed as `norm_type` or `(norm_type, parameters)`. If None, no normalization will be
        performed.\n
        `norm_type` can be any value in {`batch`, `group`, `instance`, `syncbatch`}. Please refer to PyTorch's
        [normalization layers](https://pytorch.org/docs/stable/nn.html#normalization-layers) to know the mandatory and
        optional arguments for each of them.\n
        Please note that arguments `num_channels`, `num_features` of the normalization layer
        should not be passed, as they are automatically inferred from the output of the previous layer in the network.
    dropout : Optional[float] (optional, default=None)
        dropout ratio. If None, no dropout.
    bias : bool (optional, default=True)
        whether to have a bias term in transposed convolutions.
    adn_ordering : str (optional, default="NDA")
        order of operations `Activation`, `Dropout` and `Normalization` after a transposed convolutional layer (except the
        last one).\n
        For example if "ND" is passed, `Normalization` and then `Dropout` will be performed (without `Activation`).\n
        Note: ADN will not be applied after the last convolution.

    Examples
    --------
    >>> ConvDecoder(
            in_channels=16,
            spatial_dims=2,
            channels=[8, 4, 1],
            kernel_size=(3, 5),
            stride=2,
            padding=[1, 0, 0],
            output_padding=[0, 0, (1, 2)],
            dilation=1,
            unpooling=[("upsample", {"scale_factor": 2}), ("upsample", {"size": (32, 32)})],
            unpooling_indices=[0, 1],
            act="elu",
            output_act="relu",
            norm=("batch", {"eps": 1e-05}),
            dropout=0.1,
            bias=True,
            adn_ordering="NDA",
        )
    ConvDecoder(
        (layer0): Convolution(
            (conv): ConvTranspose2d(16, 8, kernel_size=(3, 5), stride=(2, 2), padding=(1, 1))
            (adn): ADN(
                (N): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (D): Dropout(p=0.1, inplace=False)
                (A): ELU(alpha=1.0)
            )
        )
        (unpool0): Upsample(scale_factor=2.0, mode='nearest')
        (layer1): Convolution(
            (conv): ConvTranspose2d(8, 4, kernel_size=(3, 5), stride=(2, 2))
            (adn): ADN(
                (N): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (D): Dropout(p=0.1, inplace=False)
                (A): ELU(alpha=1.0)
            )
        )
        (unpool1): Upsample(size=(32, 32), mode='nearest')
        (layer2): Convolution(
            (conv): ConvTranspose2d(4, 1, kernel_size=(3, 5), stride=(2, 2), output_padding=(1, 2))
        )
        (output_act): ReLU()
    )

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: ConvParameters = 3,
        stride: ConvParameters = 1,
        padding: ConvParameters = 0,
        output_padding: ConvParameters = 0,
        dilation: ConvParameters = 1,
        unpooling: Optional[UnpoolingParameters] = (
            UnpoolingLayer.UPSAMPLE,
            {"scale_factor": 2},
        ),
        unpooling_indices: Optional[Sequence[int]] = None,
        act: Optional[ActivationParameters] = ActFunction.PRELU,
        output_act: Optional[ActivationParameters] = None,
        norm: Optional[ConvNormalizationParameters] = ConvNormLayer.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
        adn_ordering: str = "NDA",
        _input_size: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        self._current_size = _input_size if _input_size else None

        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.channels = ensure_tuple(channels)
        self.n_layers = len(self.channels)

        self.kernel_size = ensure_list_of_tuples(
            kernel_size, self.spatial_dims, self.n_layers, "kernel_size"
        )
        self.stride = ensure_list_of_tuples(
            stride, self.spatial_dims, self.n_layers, "stride"
        )
        self.padding = ensure_list_of_tuples(
            padding, self.spatial_dims, self.n_layers, "padding"
        )
        self.output_padding = ensure_list_of_tuples(
            output_padding, self.spatial_dims, self.n_layers, "output_padding"
        )
        self.dilation = ensure_list_of_tuples(
            dilation, self.spatial_dims, self.n_layers, "dilation"
        )

        self.unpooling_indices = check_pool_indices(unpooling_indices, self.n_layers)
        self.unpooling = self._check_unpool_layers(unpooling)
        self.act = act
        self.norm = check_norm_layer(norm)
        if self.norm == NormLayer.LAYER:
            raise ValueError("Layer normalization not implemented in ConvDecoder.")
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        n_unpoolings = 0
        if self.unpooling and -1 in self.unpooling_indices:
            unpooling_layer = self._get_unpool_layer(
                self.unpooling[n_unpoolings], n_channels=self.in_channels
            )
            self.add_module("init_unpool", unpooling_layer)
            n_unpoolings += 1

        echannel = self.in_channels
        for i, (c, k, s, p, o_p, d) in enumerate(
            zip(
                self.channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
            )
        ):
            conv_layer = self._get_convtranspose_layer(
                in_channels=echannel,
                out_channels=c,
                kernel_size=k,
                stride=s,
                padding=p,
                output_padding=o_p,
                dilation=d,
                is_last=(i == len(channels) - 1),
            )
            self.add_module(f"layer{i}", conv_layer)
            echannel = c  # use the output channel number as the input for the next loop
            if self.unpooling and i in self.unpooling_indices:
                unpooling_layer = self._get_unpool_layer(
                    self.unpooling[n_unpoolings], n_channels=c
                )
                self.add_module(f"unpool{i}", unpooling_layer)
                n_unpoolings += 1

        self.output_act = get_act_layer(output_act) if output_act else None

    @property
    def final_size(self):
        """
        To know the size of an image at the end of the network.
        """
        return self._current_size

    @final_size.setter
    def final_size(self, fct: Callable[[Tuple[int, ...]], Tuple[int, ...]]):
        """
        Takes as input the function used to update the current image size.
        """
        if self._current_size is not None:
            self._current_size = fct(self._current_size)

    def _get_convtranspose_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        output_padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        is_last: bool,
    ) -> Convolution:
        """
        Gets the parametrized TransposedConvolution-ADN block and updates the current output size.
        """
        self.final_size = lambda size: calculate_convtranspose_out_shape(
            size, kernel_size, stride, padding, output_padding, dilation
        )

        return Convolution(
            is_transposed=True,
            conv_only=is_last,
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

    def _get_unpool_layer(
        self, unpooling: SingleLayerUnpoolingParameters, n_channels: int
    ) -> nn.Module:
        """
        Gets the parametrized unpooling layer and updates the current output size.
        """
        unpool_layer = get_unpool_layer(
            unpooling,
            spatial_dims=self.spatial_dims,
            in_channels=n_channels,
            out_channels=n_channels,
        )
        self.final_size = lambda size: calculate_unpool_out_shape(
            unpool_mode=unpooling[0],
            in_shape=size,
            **unpool_layer.__dict__,
        )
        return unpool_layer

    @classmethod
    def _check_single_unpool_layer(
        cls, unpooling: SingleLayerUnpoolingParameters
    ) -> SingleLayerUnpoolingParameters:
        """
        Checks unpooling arguments for a single pooling layer.
        """
        if not isinstance(unpooling, tuple) or len(unpooling) != 2:
            raise ValueError(
                "unpooling must be double (or a list of doubles) with first the type of unpooling and then the parameters of "
                f"the unpooling layer in a dict. Got {unpooling}"
            )
        _ = UnpoolingLayer(unpooling[0])  # check unpooling mode
        args = unpooling[1]
        if not isinstance(args, dict):
            raise ValueError(
                f"The arguments of the unpooling layer must be passed in a dict. Got {args}"
            )

        return unpooling

    def _check_unpool_layers(
        self, unpooling: UnpoolingParameters
    ) -> UnpoolingParameters:
        """
        Checks argument unpooling.
        """
        if unpooling is None:
            return unpooling
        if isinstance(unpooling, list):
            for unpool_layer in unpooling:
                self._check_single_unpool_layer(unpool_layer)
            if len(unpooling) != len(self.unpooling_indices):
                raise ValueError(
                    "If you pass a list for unpooling, the size of that list must match "
                    f"the size of unpooling_indices. Got: unpooling={unpooling} and "
                    f"unpooling_indices={self.unpooling_indices}"
                )
        elif isinstance(unpooling, tuple):
            self._check_single_unpool_layer(unpooling)
            unpooling = (unpooling,) * len(self.unpooling_indices)
        else:
            raise ValueError(
                f"unpooling can be either None, a double (string, dictionary) or a list of such doubles. Got {unpooling}"
            )

        return unpooling
