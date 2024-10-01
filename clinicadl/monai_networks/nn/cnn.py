from collections.abc import Sequence
from typing import Any, Dict, Optional

import numpy as np
import torch.nn as nn

from .fcn_encoder import FCNEncoder
from .mlp import MLP
from .utils import check_conv_args, check_mlp_args


class CNN(nn.Sequential):
    """
    A regressor/classifier with first convolutional layers and then fully connected layers.

    This network is a simple aggregation of a Fully Convolutional Network (:py:class:`clinicadl.
    monai_networks.nn.fcn_encoder.FCNEncoder`) and a Multi Layer Perceptron (:py:class:`clinicadl.
    monai_networks.nn.mlp.MLP`).

    Parameters
    ----------
    in_shape : Sequence[int]
        sequence of integers stating the dimension of the input tensor (minus batch dimension).
    num_outputs : int
        number of variables to predict.
    conv_args : Dict[str, Any]
        the arguments for the convolutional part. The arguments are those accepted by
        :py:class:`clinicadl.monai_networks.nn.fcn_encoder.FCNEncoder`, except `in_shape`
        that is specified here. So, the only mandatory argument is `channels`.
    mlp_args : Optional[Dict[str, Any]] (optional, default=None)
        the arguments for the MLP part. The arguments are those accepted by
        :py:class:`clinicadl.monai_networks.nn.mlp.MLP`, except `in_channels` that is inferred
        from the output of the convolutional part, and `out_channels` that is set to `num_outputs`.
        So, the only mandatory argument is `hidden_channels`.\n
        If None, the MLP part will be reduced to a single linear layer.

    Examples
    --------
    # a classifier
    >>> CNN(
            in_shape=(1, 10, 10),
            num_outputs=2,
            conv_args={"channels": [2, 4], "norm": None, "act": None},
            mlp_args={"hidden_channels": [5], "act": "elu", "norm": None, "output_act": "softmax"},
        )
    CNN(
        (convolutions): FCNEncoder(
            (layer_0): Convolution(
                (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
            )
            (layer_1): Convolution(
                (conv): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1))
            )
        )
        (mlp): MLP(
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (hidden_0): Sequential(
                (linear): Linear(in_features=144, out_features=5, bias=True)
                (adn): ADN(
                    (A): ELU(alpha=1.0)
                )
            )
            (output): Sequential(
                (linear): Linear(in_features=5, out_features=2, bias=True)
                (output_act): Softmax(dim=None)
            )
        )
    )

    # a regressor
    >>> CNN(
            in_shape=(1, 10, 10),
            num_outputs=2,
            conv_args={"channels": [2, 4], "norm": None, "act": None},
        )
    CNN(
        (convolutions): FCNEncoder(
            (layer_0): Convolution(
                (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
            )
            (layer_1): Convolution(
                (conv): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1))
            )
        )
        (mlp): MLP(
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (output): Linear(in_features=144, out_features=2, bias=True)
        )
    )
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        num_outputs: int,
        conv_args: Dict[str, Any],
        mlp_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        check_conv_args(conv_args)
        check_mlp_args(mlp_args)

        self.convolutions = FCNEncoder(
            in_shape=in_shape,
            **conv_args,
        )

        n_channels = (
            conv_args["channels"][-1] if len(conv_args["channels"]) > 0 else in_shape[0]
        )
        flatten_shape = int(np.prod(self.convolutions.final_size) * n_channels)
        if mlp_args is None:
            mlp_args = {"hidden_channels": []}
        self.mlp = MLP(
            in_channels=flatten_shape,
            out_channels=num_outputs,
            **mlp_args,
        )
