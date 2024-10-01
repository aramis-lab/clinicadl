from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn

from .cnn import CNN
from .conv_encoder import ConvEncoder
from .generator import Generator
from .layers import UnpoolingLayer, UpsamplingMode
from .mlp import MLP
from .utils import (
    ActivationParameters,
    calculate_conv_out_shape,
    calculate_convtranspose_out_shape,
)


class AutoEncoder(nn.Sequential):
    """
    An autoencoder with convolutional and fully connected layers.

    The user must pass the arguments to build an encoder, from its convolutional and
    fully connected parts, and the decoder will be automatically built by taking the
    symmetrical network.

    More precisely, to build the decoder, the order of the encoding layers is reverted, convolutions are
    replaced by transposed convolutions and pooling layers are replaced by upsampling layers.
    Please note that the order of `Activation`, `Dropout` and `Normalization`, defined with the
    argument `adn_ordering` in `conv_args`, is the same for the encoder and the decoder.

    Note that an `AutoEncoder` is an aggregation of a `CNN` (:py:class:`clinicadl.monai_networks.nn.
    cnn.CNN`) and a `Generator` (:py:class:`clinicadl.monai_networks.nn.generator.Generator`).

    Parameters
    ----------
    in_shape : Sequence[int]
        sequence of integers stating the dimension of the input tensor (minus batch dimension).
    latent_size : int
        size of the latent vector.
    conv_args : Dict[str, Any]
        the arguments for the convolutional part of the encoder. The arguments are those accepted
        by :py:class:`clinicadl.monai_networks.nn.conv_encoder.ConvEncoder`, except `in_shape` that
        is specified here. So, the only mandatory argument is `channels`.
    mlp_args : Optional[Dict[str, Any]] (optional, default=None)
        the arguments for the MLP part of the encoder . The arguments are those accepted by
        :py:class:`clinicadl.monai_networks.nn.mlp.MLP`, except `in_channels` that is inferred
        from the output of the convolutional part, and `out_channels` that is set to `latent_size`.
        So, the only mandatory argument is `hidden_channels`.\n
        If None, the MLP part will be reduced to a single linear layer.
    out_channels : Optional[int] (optional, default=None)
        number of output channels. If None, the output will have the same number of channels as the
        input.
    output_act : ActivationParameters (optional, default=None)
        a potential activation layer applied to the output of the network, and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.
    upsampling_mode : Union[str, UpsamplingMode] (optional, default=UpsamplingMode.NEAREST)
        interpolation mode for upsampling (see: https://pytorch.org/docs/stable/generated/
        torch.nn.Upsample.html).

    Examples
    --------
    >>> AutoEncoder(
            in_shape=(1, 16, 16),
            latent_size=8,
            conv_args={
                "channels": [2, 4],
                "pooling_indices": [0],
                "pooling": ("avg", {"kernel_size": 2}),
            },
            mlp_args={"hidden_channels": [32], "output_act": "relu"},
            out_channels=2,
            output_act="sigmoid",
            upsampling_mode="bilinear",
        )
    AutoEncoder(
        (encoder): CNN(
            (convolutions): ConvEncoder(
                (layer_0): Convolution(
                    (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
                    (adn): ADN(
                        (N): InstanceNorm2d(2, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
                        (A): PReLU(num_parameters=1)
                    )
                )
                (pool_0): AvgPool2d(kernel_size=2, stride=2, padding=0)
                (layer_1): Convolution(
                    (conv): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1))
                )
            )
            (mlp): MLP(
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (hidden_0): Sequential(
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
                (hidden_0): Sequential(
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
                (layer_0): Convolution(
                    (conv): ConvTranspose2d(4, 4, kernel_size=(3, 3), stride=(1, 1))
                    (adn): ADN(
                        (N): InstanceNorm2d(4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
                        (A): PReLU(num_parameters=1)
                    )
                )
                (unpool_0): Upsample(size=(14, 14), mode=<UpsamplingMode.BILINEAR: 'bilinear'>)
                (layer_1): Convolution(
                    (conv): ConvTranspose2d(4, 2, kernel_size=(3, 3), stride=(1, 1))
                )
                (output_act): Sigmoid()
            )
        )
    )

    """

    def __init__(
        self,
        in_shape: Sequence[int],
        latent_size: int,
        conv_args: Dict[str, Any],
        mlp_args: Optional[Dict[str, Any]] = None,
        out_channels: Optional[int] = None,
        output_act: ActivationParameters = None,
        upsampling_mode: Union[str, UpsamplingMode] = UpsamplingMode.NEAREST,
    ) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.upsampling_mode = self._check_upsampling_mode(upsampling_mode)
        self.out_channels = out_channels if out_channels else self.in_shape[0]
        self.output_act_type = output_act

        self.encoder = CNN(
            in_shape=self.in_shape,
            num_outputs=latent_size,
            conv_args=conv_args,
            mlp_args=mlp_args,
        )
        inter_channels = (
            conv_args["channels"][-1] if len(conv_args["channels"]) > 0 else in_shape[0]
        )
        inter_shape = (inter_channels, *self.encoder.convolutions.final_size)
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
        args["hidden_channels"] = cls._invert_list_arg(mlp.hidden_channels)

        return args

    def _invert_conv_args(
        self, args: Dict[str, Any], conv: ConvEncoder
    ) -> Dict[str, Any]:
        """
        Inverts arguments passed for the convolutional part of the encoder, to get the convolutional
        part of the decoder.
        """
        args["channels"] = self._invert_list_arg(conv.channels)[:-1] + [
            self.out_channels
        ]

        args["kernel_size"] = self._invert_list_arg(conv.kernel_size)
        args["stride"] = self._invert_list_arg(conv.stride)
        args["padding"] = self._invert_list_arg(conv.padding)
        args["dilation"] = self._invert_list_arg(conv.dilation)
        args["output_padding"] = self._get_output_padding_list(conv)

        args["unpooling_indices"] = (
            conv.n_layers - np.array(conv.pooling_indices) - 2
        ).astype(int)
        args["unpooling"] = []
        for size_before_pool in conv.size_before_pool[::-1]:
            args["unpooling"].append(self._invert_pooling_layer(size_before_pool))

        if "pooling" in args:
            del args["pooling"]
        if "pooling_indices" in args:
            del args["pooling_indices"]

        args["output_act"] = self.output_act_type if self.output_act_type else None

        return args

    @classmethod
    def _invert_list_arg(cls, arg: Union[Any, List[Any]]) -> Union[Any, List[Any]]:
        """
        Reverses lists.
        """
        return list(arg[::-1]) if isinstance(arg, Sequence) else arg

    def _invert_pooling_layer(
        self, size_before_pool: Sequence[int]
    ) -> Tuple[UnpoolingLayer, Dict[str, Any]]:
        """
        Gets the unpooling layer (always upsample).
        """
        return (
            UnpoolingLayer.UPSAMPLE,
            {"size": size_before_pool, "mode": self.upsampling_mode},
        )

    @classmethod
    def _get_output_padding_list(cls, conv: ConvEncoder) -> List[Tuple[int, ...]]:
        """
        Finds output padding list.
        """
        output_padding = []
        size_before_conv = (
            conv.size_before_conv if len(conv.size_before_conv) > 0 else [conv.in_shape]
        )
        for size, k, s, p, d in zip(
            size_before_conv, conv.kernel_size, conv.stride, conv.padding, conv.dilation
        ):
            out_p = cls._find_output_padding(size, k, s, p, d)
            output_padding.append(out_p)

        return cls._invert_list_arg(output_padding)

    @classmethod
    def _find_output_padding(
        cls,
        in_shape: Union[Sequence[int], int],
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        padding: Union[Sequence[int], int],
        dilation: Union[Sequence[int], int],
    ) -> Tuple[int, ...]:
        """
        Finds output padding necessary to recover the right image size after
        a transposed convolution.
        """
        in_shape_np = np.atleast_1d(in_shape)
        conv_out_shape = calculate_conv_out_shape(
            in_shape_np, kernel_size, stride, padding, dilation
        )
        convt_out_shape = calculate_convtranspose_out_shape(
            conv_out_shape, kernel_size, stride, padding, 0, dilation
        )
        output_padding = in_shape_np - np.atleast_1d(convt_out_shape)

        return tuple(int(s) for s in output_padding)

    def _check_upsampling_mode(
        self, upsampling_mode: Union[str, UpsamplingMode]
    ) -> UpsamplingMode:
        """
        Checks consistency between data shape and upsampling mode.
        """
        upsampling_mode = UpsamplingMode(upsampling_mode)
        if upsampling_mode == "linear" and len(self.in_shape) != 2:
            raise ValueError(
                f"upsampling mode `linear` only works with 2D data (counting the channel dimension). "
                f"Got in_shape={self.in_shape}, which is understood as {len(self.in_shape)}D data."
            )
        elif upsampling_mode == "bilinear" and len(self.in_shape) != 3:
            raise ValueError(
                f"upsampling mode `bilinear` only works with 3D data (counting the channel dimension). "
                f"Got in_shape={self.in_shape}, which is understood as {len(self.in_shape)}D data."
            )
        elif upsampling_mode == "bicubic" and len(self.in_shape) != 3:
            raise ValueError(
                f"upsampling mode `bicubic` only works with 3D data (counting the channel dimension). "
                f"Got in_shape={self.in_shape}, which is understood as {len(self.in_shape)}D data."
            )
        elif upsampling_mode == "trilinear" and len(self.in_shape) != 4:
            raise ValueError(
                f"upsampling mode `trilinear` only works with 4D data (counting the channel dimension). "
                f"Got in_shape={self.in_shape}, which is understood as {len(self.in_shape)}D data."
            )

        return upsampling_mode
