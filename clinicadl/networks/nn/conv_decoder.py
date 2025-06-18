from typing import Callable, List, Optional, Sequence, Tuple

import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_act_layer

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
    check_adn_ordering,
    check_norm_layer,
    check_pool_indices,
    ensure_list_of_tuples,
)


class ConvDecoder(nn.Sequential):
    """
    Fully convolutional decoder network with transposed convolutions, unpooling, normalization, activation
    and dropout layers.

    It is the symmetric of :py:class:`~clinicadl.networks.nn.ConvEncoder`, where convolutions are replaced
    by transposed convolutions, and pooling layers by unpooling layers.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of channels in the input image.
    channels : Sequence[int]
        Number of output channels of each transposed convolution. Thus, this
        parameter also controls the number of transposed convolutions (equal to the length of the sequence).
    kernel_size : ConvParameters, default=3
        Kernel size of the transposed convolutions. Can be an ``int``, a ``tuple``, or a ``list``:

        - ``int``: the value will be used for all layers and all dimensions;
        - ``tuple`` (e.g. ``(3, 3, 2)``): it will be interpreted as the values for each dimension. These values
          will be used for all the layers;
        - ``list`` (e.g. ``[(3, 3, 2), 3]``): it will be interpreted as the kernel sizes for each layer.
          The length of the list must be equal to the number of transposed convolutions (i.e. ``len(channels)``).
    stride : ConvParameters, default=1
        Stride of the transposed convolutions. Can be an ``int``, a ``tuple``, or a ``list``, and is passed in the same way
        as ``kernel_size``.\n
    padding : ConvParameters, default=0
        Padding of the transposed convolutions. Can be an ``int``, a ``tuple``, or a ``list``, and is passed in the same way
        as ``kernel_size``.\n
    output_padding : ConvParameters, default=0
        Output padding of the transposed convolutions. Can be an ``int``, a ``tuple``, or a ``list``, and is passed in the same way
        as ``kernel_size``.\n
    dilation : ConvParameters, default=1
        Dilation factor of the transposed convolutions. Can be an ``int``, a ``tuple``, or a ``list``, and is passed in the same way
        as ``kernel_size``.\n
    unpooling : Optional[UnpoolingParameters], default=("upsample", {"scale_factor": 2})
        The unpooling mode and the arguments of the unpooling layer, passed as ``(unpooling_mode, arguments)``,  where ``arguments`` is a dictionary.
        If ``None``, no unpooling will be performed in the network.\n
        ``unpooling_mode`` can be either ``upsample`` or ``convtranspose``. Please refer to :py:class:`torch.nn.Upsample`
        or :py:class:`torch.nn.ConvTranspose3d` to know their arguments.\n
        If a ``list`` is passed, it will be understood as the unpooling for each unpooling layer.

        .. note::
            No need to pass ``in_channels`` and ``out_channels`` for ``convtranspose``, because the unpooling
            layers are not intended to modify the number of channels here.

    unpooling_indices : Optional[Sequence[int]], default=None
        Indices of the transposed convolutions after which unpooling should be performed.
        If ``None``, no unpooling will be performed. An index equal to ``-1`` will be understood as a pooling layer before
        the first transposed convolution.
    act : Optional[ActivationParameters], default="prelu"
        The activation function used after a transposed convolution, and optionally its arguments.
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
        The normalization layer used after a transposed convolution, and optionally its arguments.
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
        If a ``list`` is passed for ``kernel_size``, ``stride``, ``padding``, ``output_padding``, or ``dilation``, and the size of this
        list in not equal to ``len(channels)``.
    ValueError
        If indices in ``unpooling_indices`` are greater than ``len(channels)-1`` (``len(channels)-1`` being the index of the last
        transposed convolution).
    ValueError
        If a ``list`` is passed for ``unpooling``, and ``len(unpooling)!=len(unpooling_indices)``.
    ValueError
        If the activation or normalization layer requires a mandatory argument, which is not passed by the user (via a dictionary
        in ``act`` or ``norm``).

    Examples
    --------

    .. code-block:: python

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

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ConvEncoder`
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

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
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
        self.unpooling = check_unpool_layers(
            unpooling, unpooling_indices=self.unpooling_indices
        )
        self.act = act
        self.norm = check_norm_layer(norm)
        if self.norm == NormLayer.LAYER:
            raise ValueError("Layer normalization not implemented in ConvDecoder.")
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = check_adn_ordering(adn_ordering)

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
    def _final_size(self):
        """
        To know the size of an image at the end of the network.
        """
        return self._current_size

    @_final_size.setter
    def _final_size(self, fct: Callable[[Tuple[int, ...]], Tuple[int, ...]]):
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
        self._final_size = lambda size: calculate_convtranspose_out_shape(
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
        self._final_size = lambda size: calculate_unpool_out_shape(
            unpool_mode=unpooling[0],
            in_shape=size,
            **unpool_layer.__dict__,
        )
        return unpool_layer


def check_unpool_layers(
    unpooling: UnpoolingParameters, unpooling_indices: Sequence[int]
) -> List[SingleLayerUnpoolingParameters]:
    """
    Checks argument unpooling.
    """
    if unpooling is None:
        return unpooling
    if isinstance(unpooling, list):
        for unpool_layer in unpooling:
            _check_single_unpool_layer(unpool_layer)
        if len(unpooling) != len(unpooling_indices):
            raise ValueError(
                "If you pass a list for unpooling, the size of that list must match "
                f"the size of unpooling_indices. Got: unpooling={unpooling} and "
                f"unpooling_indices={unpooling_indices}"
            )
    elif isinstance(unpooling, tuple):
        _check_single_unpool_layer(unpooling)
        unpooling = [unpooling] * len(unpooling_indices)
    else:
        raise ValueError(
            f"unpooling can be either None, a double (string, dictionary) or a list of such doubles. Got {unpooling}"
        )

    return unpooling


def _check_single_unpool_layer(unpooling: SingleLayerUnpoolingParameters) -> None:
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
