from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch.nn as nn

from .conv_encoder import ConvEncoder
from .mlp import MLP
from .utils import check_conv_args, check_mlp_args


class CNN(nn.Sequential):
    """
    A regressor/classifier with first convolutional layers and then fully connected layers.

    This network is a simple aggregation of a :py:class:`~clinicadl.networks.nn.ConvEncoder`
    and a :py:class:`~clinicadl.networks.nn.MLP`.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    Parameters
    ----------
    in_shape : Sequence[int]
        Dimensions of the input tensor (without batch dimension).
    num_outputs : int
        Number of variables to predict.
    conv_args : Dict[str, Any]
        The arguments for the convolutional part. The arguments are those accepted by
        :py:class:`~clinicadl.networks.nn.ConvEncoder`, except ``spatial_dims`` and ``in_channels``
        that are specified here via ``in_shape``. So, the only **mandatory argument is** ``channels``.
    mlp_args : Optional[Dict[str, Any]], default=None
        The arguments for the MLP part. The arguments are those accepted by
        :py:class:`~clinicadl.networks.nn.MLP`, except ``num_inputs`` that is inferred
        from the output of the convolutional part, and ``num_outputs`` that is set here.
        So, the only **mandatory argument is** ``hidden_dims``.\n
        If ``None``, the MLP part will be reduced to a single linear layer.

    Raises
    ------
    ValueError
        If ``conv_args`` doesn't contain the key ``channels``.
    ValueError
        If ``mlp_args`` is not ``None`` and doesn't contain the key ``hidden_dims``.

    Examples
    --------

    .. code-block:: python

        # a classifier
        >>> CNN(
                in_shape=(1, 10, 10),
                num_outputs=2,
                conv_args={"channels": [2, 4], "norm": None, "act": None},
                mlp_args={"hidden_dims": [5], "act": "elu", "norm": None, "output_act": "softmax"},
            )
        CNN(
            (convolutions): ConvEncoder(
                (layer0): Convolution(
                    (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
                )
                (layer1): Convolution(
                    (conv): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1))
                )
            )
            (mlp): MLP(
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (hidden0): Sequential(
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

    .. code-block:: python

        # a regressor
        >>> CNN(
                in_shape=(1, 10, 10),
                num_outputs=2,
                conv_args={"channels": [2, 4], "norm": None, "act": None},
            )
        CNN(
            (convolutions): ConvEncoder(
                (layer0): Convolution(
                    (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
                )
                (layer1): Convolution(
                    (conv): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1))
                )
            )
            (mlp): MLP(
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (output): Linear(in_features=144, out_features=2, bias=True)
            )
        )

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ConvEncoder`
    :py:class:`~clinicadl.networks.nn.MLP`
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
        self.in_shape = in_shape
        self.num_outputs = num_outputs

        in_channels, *input_size = in_shape
        spatial_dims = len(input_size)

        self.convolutions = ConvEncoder(
            in_channels=in_channels,
            spatial_dims=spatial_dims,
            _input_size=tuple(input_size),
            **conv_args,
        )

        n_channels = (
            conv_args["channels"][-1] if len(conv_args["channels"]) > 0 else in_shape[0]
        )
        flatten_shape = int(np.prod(self.convolutions._final_size) * n_channels)
        if mlp_args is None:
            mlp_args = {"hidden_dims": []}
        self.mlp = MLP(
            num_inputs=flatten_shape,
            num_outputs=num_outputs,
            **mlp_args,
        )
