from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch.nn as nn
from monai.networks.layers.simplelayers import Reshape

from .conv_decoder import ConvDecoder
from .mlp import MLP
from .utils import check_conv_args, check_mlp_args


class Generator(nn.Sequential):
    """
    A generator with first fully-connected layers and then convolutional layers.

    This network is a simple aggregation of a :py:class:`~clinicadl.networks.nn.MLP`
    and a :py:class:`~clinicadl.networks.nn.ConvDecoder`.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    Parameters
    ----------
    latent_size : int
        Size of the latent vector.
    start_shape : Sequence[int]
        Initial shape of the image, i.e. the shape at the
        beginning of the convolutional part (without batch dimension, but including the channel dimension).\n
        Thus, ``start_shape`` also determines the dimension of the output of the generator (the exact
        shape depends on the convolutional part and can be accessed via the attribute
        ``output_shape``).
    conv_args : Dict[str, Any]
        The arguments for the convolutional part. The arguments are those accepted by
        :py:class:`~clinicadl.networks.nn.ConvDecoder`, except ``spatial_dims`` and ``in_channels``
        that are specified here via ``start_shape``. So, the only **mandatory argument is** ``channels``.
    mlp_args : Optional[Dict[str, Any]], default=None
        The arguments for the MLP part. The arguments are those accepted by
        :py:class:`~clinicadl.networks.nn.MLP`, except ``num_inputs`` that is equal here to
        ``latent_size``, and ``num_outputs`` that is inferred here from ``start_shape``.
        So, the only **mandatory argument is** ``hidden_dims``.\n
        If ``None``, the MLP part will be reduced to a single linear layer.

    Attributes
    ----------
    output_shape : int
        The shape of the output image, computed from ``start_shape``.

    Raises
    ------
    ValueError
        If ``conv_args`` doesn't contain the key ``channels``.
    ValueError
        If ``mlp_args`` is not ``None`` and doesn't contain the key ``hidden_dims``.

    Examples
    --------

    .. code-block:: python

        >>> Generator(
                latent_size=8,
                start_shape=(8, 2, 2),
                conv_args={"channels": [4, 2], "norm": None, "act": None},
                mlp_args={"hidden_dims": [16], "act": "elu", "norm": None},
            )
        Generator(
            (mlp): MLP(
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (hidden0): Sequential(
                    (linear): Linear(in_features=8, out_features=16, bias=True)
                    (adn): ADN(
                        (A): ELU(alpha=1.0)
                    )
                )
                (output): Linear(in_features=16, out_features=32, bias=True)
            )
            (reshape): Reshape()
            (convolutions): ConvDecoder(
                (layer0): Convolution(
                    (conv): ConvTranspose2d(8, 4, kernel_size=(3, 3), stride=(1, 1))
                )
                (layer1): Convolution(
                    (conv): ConvTranspose2d(4, 2, kernel_size=(3, 3), stride=(1, 1))
                )
            )
        )

        >>> Generator(
                latent_size=8,
                start_shape=(8, 2, 2),
                conv_args={"channels": [4, 2], "norm": None, "act": None, "output_act": "relu"},
            )
        Generator(
            (mlp): MLP(
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (output): Linear(in_features=8, out_features=32, bias=True)
            )
            (reshape): Reshape()
            (convolutions): ConvDecoder(
                (layer0): Convolution(
                    (conv): ConvTranspose2d(8, 4, kernel_size=(3, 3), stride=(1, 1))
                )
                (layer1): Convolution(
                    (conv): ConvTranspose2d(4, 2, kernel_size=(3, 3), stride=(1, 1))
                )
                (output_act): ReLU()
            )
        )

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ConvDecoder`
    :py:class:`~clinicadl.networks.nn.MLP`
    """

    def __init__(
        self,
        latent_size: int,
        start_shape: Sequence[int],
        conv_args: Dict[str, Any],
        mlp_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        check_conv_args(conv_args)
        check_mlp_args(mlp_args)
        self.latent_size = latent_size
        self.start_shape = start_shape

        flatten_shape = int(np.prod(start_shape))
        if mlp_args is None:
            mlp_args = {"hidden_dims": []}
        self.mlp = MLP(
            num_inputs=latent_size,
            num_outputs=flatten_shape,
            **mlp_args,
        )

        self.reshape = Reshape(*start_shape)
        inter_channels, *inter_size = start_shape
        self.convolutions = ConvDecoder(
            in_channels=inter_channels,
            spatial_dims=len(inter_size),
            _input_size=inter_size,
            **conv_args,
        )

        n_channels = (
            conv_args["channels"][-1]
            if len(conv_args["channels"]) > 0
            else start_shape[0]
        )
        self.output_shape = (n_channels, *self.convolutions._final_size)
