from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch.nn as nn

from .cnn import CNN
from .conv_encoder import ConvEncoder
from .generator import Generator
from .layers.utils import (
    ActivationParameters,
    PoolingLayer,
    SingleLayerPoolingParameters,
    SingleLayerUnpoolingParameters,
    UnpoolingLayer,
    UnpoolingMode,
)
from .mlp import MLP
from .utils import (
    calculate_conv_out_shape,
    calculate_convtranspose_out_shape,
    calculate_pool_out_shape,
)


class AutoEncoder(nn.Sequential):
    """
    An AutoEncoder with convolutional and fully connected layers.

    The user must pass the arguments to build an encoder, from its convolutional and
    fully connected parts, and the decoder will be automatically built by taking the
    symmetrical network.

    More precisely, to build the decoder, the order of the encoding layers is reverted, convolutions are
    replaced by transposed convolutions, and pooling layers are replaced by either upsampling or transposed
    convolution layers.

    An ``AutoEncoder`` is an aggregation of a :py:class:`~clinicadl.networks.nn.CNN` and a
    :py:class:`~clinicadl.networks.nn.Generator`.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    .. note::
        Please note that the order of Activation, Dropout and Normalization, defined with the
        argument ``adn_ordering`` in ``conv_args``, is the same for the encoder and the decoder.

    Parameters
    ----------
    in_shape : Sequence[int]
        Dimensions of the input tensor (without batch dimension).
    latent_size : int
        Size of the latent vector.
    conv_args : Dict[str, Any]
        The arguments for the convolutional part. The arguments are those accepted by
        :py:class:`~clinicadl.networks.nn.ConvEncoder`, except ``spatial_dims`` and ``in_channels``
        that are specified here via ``in_shape``. So, the only **mandatory argument is** ``channels``.
    mlp_args : Optional[Dict[str, Any]], default=None
        The arguments for the MLP part. The arguments are those accepted by
        :py:class:`~clinicadl.networks.nn.MLP`, except ``num_inputs`` that is inferred
        from the output of the convolutional part, and ``num_outputs`` that is equal to ``latent_size`` here.
        So, the only **mandatory argument is** ``hidden_dims``.\n
        If ``None``, the MLP part will be reduced to a single linear layer.
    out_channels : Optional[int], default=None
        Number of output channels. If ``None``, the output will have the same number of channels as the
        input.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.
    unpooling_mode : Union[str, UnpoolingMode], default=UnpoolingMode.NEAREST
        Type of unpooling. Can be any value in {``nearest``, ``linear``, ``bilinear``, ``bicubic``, ``trilinear`` or
        ``convtranspose``}:

        - ``nearest``: unpooling is performed by upsampling with the `nearest` algorithm (see
          :py:class:`torch.nn.Upsample`);
        - ``linear``: unpooling is performed by upsampling with the `linear` algorithm. Only works with 1D images (excluding the
          channel dimension);
        - ``bilinear``: unpooling is performed by upsampling with the `bilinear` algorithm. Only works with 2D images;
        - ``bicubic``: unpooling is performed by upsampling with the `bicubic` algorithm. Only works with 2D images;
        - ``trilinear``: unpooling is performed by upsampling with the `trilinear` algorithm. Only works with 3D images;
        - ``convtranspose``: unpooling is performed with a transposed convolution (see :py:class:`torch.nn.ConvTranspose3d`), whose
          parameters (kernel size, stride, etc.) are computed to reverse the pooling operation.

    Examples
    --------

    .. code-block:: python

        >>> AutoEncoder(
                in_shape=(1, 16, 16),
                latent_size=8,
                conv_args={
                    "channels": [2, 4],
                    "pooling_indices": [0],
                    "pooling": ("avg", {"kernel_size": 2}),
                },
                mlp_args={"hidden_dims": [32], "output_act": "relu"},
                out_channels=2,
                output_act="sigmoid",
                unpooling_mode="bilinear",
            )
        AutoEncoder(
            (encoder): CNN(
                (convolutions): ConvEncoder(
                    (layer0): Convolution(
                        (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
                        (adn): ADN(
                            (N): InstanceNorm2d(2, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
                            (A): PReLU(num_parameters=1)
                        )
                    )
                    (pool0): AvgPool2d(kernel_size=2, stride=2, padding=0)
                    (layer1): Convolution(
                        (conv): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1))
                    )
                )
                (mlp): MLP(
                    (flatten): Flatten(start_dim=1, end_dim=-1)
                    (hidden0): Sequential(
                        (linear): Linear(in_features=100, out_features=32, bias=True)
                        (adn): ADN(
                            (N): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                            (A): PReLU(num_parameters=1)
                        )
                    )
                    (output): Sequential(
                        (linear): Linear(in_features=32, out_features=8, bias=True)
                        (output_act): ReLU()
                    )
                )
            )
            (decoder): Generator(
                (mlp): MLP(
                    (flatten): Flatten(start_dim=1, end_dim=-1)
                    (hidden0): Sequential(
                        (linear): Linear(in_features=8, out_features=32, bias=True)
                        (adn): ADN(
                            (N): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                            (A): PReLU(num_parameters=1)
                        )
                    )
                    (output): Sequential(
                        (linear): Linear(in_features=32, out_features=100, bias=True)
                        (output_act): ReLU()
                    )
                )
                (reshape): Reshape()
                (convolutions): ConvDecoder(
                    (layer0): Convolution(
                        (conv): ConvTranspose2d(4, 4, kernel_size=(3, 3), stride=(1, 1))
                        (adn): ADN(
                            (N): InstanceNorm2d(4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
                            (A): PReLU(num_parameters=1)
                        )
                    )
                    (unpool0): Upsample(size=(14, 14), mode=<UpsamplingMode.BILINEAR: 'bilinear'>)
                    (layer1): Convolution(
                        (conv): ConvTranspose2d(4, 2, kernel_size=(3, 3), stride=(1, 1))
                    )
                    (output_act): Sigmoid()
                )
            )
        )

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.CNN`
    :py:class:`~clinicadl.networks.nn.Generator`
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        latent_size: int,
        conv_args: Dict[str, Any],
        mlp_args: Optional[Dict[str, Any]] = None,
        out_channels: Optional[int] = None,
        output_act: Optional[ActivationParameters] = None,
        unpooling_mode: Union[str, UnpoolingMode] = UnpoolingMode.NEAREST,
    ) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.latent_size = latent_size
        self.out_channels = out_channels if out_channels else self.in_shape[0]
        self._output_act = output_act
        self.spatial_dims = len(in_shape[1:])
        self.unpooling_mode = check_unpooling_mode(unpooling_mode, self.spatial_dims)

        self.encoder = CNN(
            in_shape=self.in_shape,
            num_outputs=latent_size,
            conv_args=conv_args,
            mlp_args=mlp_args,
        )
        inter_channels = (
            conv_args["channels"][-1] if len(conv_args["channels"]) > 0 else in_shape[0]
        )
        inter_shape = (inter_channels, *self.encoder.convolutions._final_size)
        self.decoder = Generator(
            latent_size=latent_size,
            start_shape=inter_shape,
            conv_args=self._invert_conv_args(conv_args, self.encoder.convolutions),
            mlp_args=self._invert_mlp_args(mlp_args, self.encoder.mlp),
        )

    @classmethod
    def _invert_mlp_args(cls, args: Dict[str, Any], mlp: MLP) -> Dict[str, Any]:
        """
        Inverts arguments passed for the MLP part of the encoder, to get the MLP part of
        the decoder.
        """
        if args is None:
            args = {}
        args["hidden_dims"] = cls._invert_list_arg(mlp.hidden_dims)

        return args

    def _invert_conv_args(
        self, args: Dict[str, Any], conv: ConvEncoder
    ) -> Dict[str, Any]:
        """
        Inverts arguments passed for the convolutional part of the encoder, to get the convolutional
        part of the decoder.
        """
        if len(args["channels"]) == 0:
            args["channels"] = []
        else:
            args["channels"] = self._invert_list_arg(conv.channels[:-1]) + [
                self.out_channels
            ]
        args["kernel_size"] = self._invert_list_arg(conv.kernel_size)
        args["stride"] = self._invert_list_arg(conv.stride)
        args["dilation"] = self._invert_list_arg(conv.dilation)
        args["padding"], args["output_padding"] = self._get_paddings_list(conv)

        args["unpooling_indices"] = list(
            (conv.n_layers - np.array(conv.pooling_indices) - 2).astype(int)
        )
        args["unpooling"] = []
        sizes_before_pooling = [
            size
            for size, (layer_name, _) in zip(conv._size_details, conv.named_children())
            if "pool" in layer_name
        ]
        for size, pooling in zip(sizes_before_pooling[::-1], conv.pooling[::-1]):
            args["unpooling"].append(self._invert_pooling_layer(size, pooling))

        if "pooling" in args:
            del args["pooling"]
        if "pooling_indices" in args:
            del args["pooling_indices"]

        args["output_act"] = self._output_act if self._output_act else None

        return args

    @classmethod
    def _invert_list_arg(cls, arg: Union[Any, List[Any]]) -> Union[Any, List[Any]]:
        """
        Reverses lists.
        """
        return list(arg[::-1]) if isinstance(arg, Sequence) else arg

    def _invert_pooling_layer(
        self,
        size_before_pool: Sequence[int],
        pooling: SingleLayerPoolingParameters,
    ) -> SingleLayerUnpoolingParameters:
        """
        Gets the unpooling layer.
        """
        if self.unpooling_mode == UnpoolingMode.CONV_TRANS:
            return (
                UnpoolingLayer.CONV_TRANS,
                self._invert_pooling_with_convtranspose(size_before_pool, pooling),
            )
        else:
            return (
                UnpoolingLayer.UPSAMPLE,
                {"size": size_before_pool, "mode": self.unpooling_mode},
            )

    @classmethod
    def _invert_pooling_with_convtranspose(
        cls,
        size_before_pool: Sequence[int],
        pooling: SingleLayerPoolingParameters,
    ) -> Dict[str, Any]:
        """
        Computes the arguments of the transposed convolution, based on the pooling layer.
        """
        pooling_mode, pooling_args = pooling
        if (
            pooling_mode == PoolingLayer.ADAPT_AVG
            or pooling_mode == PoolingLayer.ADAPT_MAX
        ):
            input_size_np = np.array(size_before_pool)
            output_size_np = np.array(pooling_args["output_size"])
            stride_np = input_size_np // output_size_np  # adaptive pooling formulas
            kernel_size_np = (
                input_size_np - (output_size_np - 1) * stride_np
            )  # adaptive pooling formulas
            args = {
                "kernel_size": tuple(int(k) for k in kernel_size_np),
                "stride": tuple(int(s) for s in stride_np),
            }
            padding, output_padding = cls._find_convtranspose_paddings(
                pooling_mode,
                size_before_pool,
                output_size=pooling_args["output_size"],
                **args,
            )

        elif pooling_mode == PoolingLayer.MAX or pooling_mode == PoolingLayer.AVG:
            if "stride" not in pooling_args:
                pooling_args["stride"] = pooling_args["kernel_size"]
            args = {
                arg: value
                for arg, value in pooling_args.items()
                if arg in ["kernel_size", "stride", "padding", "dilation"]
            }
            padding, output_padding = cls._find_convtranspose_paddings(
                pooling_mode,
                size_before_pool,
                **pooling_args,
            )

        args["padding"] = padding  # pylint: disable=possibly-used-before-assignment
        args["output_padding"] = output_padding  # pylint: disable=possibly-used-before-assignment

        return args

    @classmethod
    def _get_paddings_list(cls, conv: ConvEncoder) -> List[Tuple[int, ...]]:
        """
        Finds output padding list.
        """
        padding = []
        output_padding = []
        size_before_convs = [
            size
            for size, (layer_name, _) in zip(conv._size_details, conv.named_children())
            if "layer" in layer_name
        ]
        for size, k, s, p, d in zip(
            size_before_convs,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
        ):
            p, out_p = cls._find_convtranspose_paddings(
                "conv", size, kernel_size=k, stride=s, padding=p, dilation=d
            )
            padding.append(p)
            output_padding.append(out_p)

        return cls._invert_list_arg(padding), cls._invert_list_arg(output_padding)

    @classmethod
    def _find_convtranspose_paddings(
        cls,
        layer_type: Union[Literal["conv"], PoolingLayer],
        in_shape: Union[Sequence[int], int],
        padding: Union[Sequence[int], int] = 0,
        **kwargs,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Finds padding and output padding necessary to recover the right image size after
        a transposed convolution.
        """
        if layer_type == "conv":
            layer_out_shape = calculate_conv_out_shape(in_shape, **kwargs)
        elif layer_type in list(PoolingLayer):
            layer_out_shape = calculate_pool_out_shape(layer_type, in_shape, **kwargs)

        convt_out_shape = calculate_convtranspose_out_shape(layer_out_shape, **kwargs)  # pylint: disable=possibly-used-before-assignment
        output_padding = np.atleast_1d(in_shape) - np.atleast_1d(convt_out_shape)

        if (
            output_padding < 0
        ).any():  # can happen with ceil_mode=True for maxpool. Then, add some padding
            padding = np.atleast_1d(padding) * np.ones_like(
                output_padding
            )  # to have the same shape as output_padding
            padding[output_padding < 0] += np.maximum(np.abs(output_padding) // 2, 1)[
                output_padding < 0
            ]  # //2 because 2*padding pixels are removed

            convt_out_shape = calculate_convtranspose_out_shape(
                layer_out_shape, padding=padding, **kwargs
            )
            output_padding = np.atleast_1d(in_shape) - np.atleast_1d(convt_out_shape)
            padding = tuple(int(s) for s in padding)

        return padding, tuple(int(s) for s in output_padding)


def check_unpooling_mode(
    unpooling_mode: Union[str, UnpoolingMode], dim: int
) -> UnpoolingMode:
    """
    Checks consistency between data shape and unpooling mode.
    """
    unpooling_mode = UnpoolingMode(unpooling_mode)
    if unpooling_mode == UnpoolingMode.LINEAR and dim != 1:
        raise ValueError(
            f"unpooling mode `linear` only works with 1D data (spatial dimensions). "
            f"Got {dim}D data."
        )
    elif unpooling_mode == UnpoolingMode.BILINEAR and dim != 2:
        raise ValueError(
            f"unpooling mode `bilinear` only works with 2D data (spatial dimensions). "
            f"Got {dim}D data."
        )
    elif unpooling_mode == UnpoolingMode.BICUBIC and dim != 2:
        raise ValueError(
            f"unpooling mode `bicubic` only works with 2D data (spatial dimensions). "
            f"Got {dim}D data."
        )
    elif unpooling_mode == UnpoolingMode.TRILINEAR and dim != 3:
        raise ValueError(
            f"unpooling mode `trilinear` only works with 3D data (spatial dimensions). "
            f"Got {dim}D data."
        )

    return unpooling_mode
