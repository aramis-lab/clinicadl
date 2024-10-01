from collections.abc import Sequence
from typing import Any, Dict, Optional

import numpy as np
import torch.nn as nn
from monai.networks.layers.simplelayers import Reshape

from .fcn_decoder import FCNDecoder
from .mlp import MLP
from .utils import check_conv_args, check_mlp_args


class Generator(nn.Sequential):
    """
    A generator with first fully connected layers and then convolutional layers.

    This network is a simple aggregation of a Multi Layer Perceptron (:py:class:
    `clinicadl.monai_networks.nn.mlp.MLP`) and a Fully Convolutional Network
    (:py:class:`clinicadl.monai_networks.nn.fcn_decoder.FCNDecoder`).

    Parameters
    ----------
    latent_size : int
        size of the latent vector.
    start_shape : Sequence[int]
        sequence of integers stating the initial shape of the image, i.e. the shape at the
        beginning of the convolutional part.\n
        Thus, `start_shape` determines the dimension of the output of the generator (the exact
        shape depends on the convolutional part and can be accessed via the class attribute
        `output_shape`).
    conv_args : Dict[str, Any]
        the arguments for the convolutional part. The arguments are those accepted by
        :py:class:`clinicadl.monai_networks.nn.fcn_decoder.FCNDecoder`, except `in_shape` that
        is specified here via `start_shape`. So, the only mandatory argument is `channels`.
    mlp_args : Optional[Dict[str, Any]] (optional, default=None)
        the arguments for the MLP part. The arguments are those accepted by
        :py:class:`clinicadl.monai_networks.nn.mlp.MLP`, except `in_channels` that is specified
        here via `latent_size`, and `out_channels` that is inferred from `start_shape`.
        So, the only mandatory argument is `hidden_channels`.\n
        If None, the MLP part will be reduced to a single linear layer.

    Examples
    --------
    >>> Generator(
            latent_size=8,
            start_shape=(8, 2, 2),
            conv_args={"channels": [4, 2], "norm": None, "act": None},
            mlp_args={"hidden_channels": [16], "act": "elu", "norm": None},
        )
    Generator(
        (mlp): MLP(
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (hidden_0): Sequential(
                (linear): Linear(in_features=8, out_features=16, bias=True)
                (adn): ADN(
                    (A): ELU(alpha=1.0)
                )
            )
            (output): Linear(in_features=16, out_features=32, bias=True)
        )
        (reshape): Reshape()
        (convolutions): FCNDecoder(
            (layer_0): Convolution(
                (conv): ConvTranspose2d(8, 4, kernel_size=(3, 3), stride=(1, 1))
            )
            (layer_1): Convolution(
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
        (convolutions): FCNDecoder(
            (layer_0): Convolution(
                (conv): ConvTranspose2d(8, 4, kernel_size=(3, 3), stride=(1, 1))
            )
            (layer_1): Convolution(
                (conv): ConvTranspose2d(4, 2, kernel_size=(3, 3), stride=(1, 1))
            )
            (output_act): ReLU()
        )
    )
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

        flatten_shape = int(np.prod(start_shape))
        if mlp_args is None:
            mlp_args = {"hidden_channels": []}
        self.mlp = MLP(
            in_channels=latent_size,
            out_channels=flatten_shape,
            **mlp_args,
        )

        self.reshape = Reshape(*start_shape)
        self.convolutions = FCNDecoder(
            in_shape=start_shape,
            **conv_args,
        )

        n_channels = (
            conv_args["channels"][-1]
            if len(conv_args["channels"]) > 0
            else start_shape[0]
        )
        self.output_shape = (n_channels, *self.convolutions.final_size)
