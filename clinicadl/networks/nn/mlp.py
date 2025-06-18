from collections import OrderedDict
from typing import Optional, Sequence

import torch.nn as nn
from monai.networks.blocks import ADN
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets import FullyConnectedNet as BaseMLP

from .layers.utils import (
    ActFunction,
    ActivationParameters,
    NormalizationParameters,
    NormLayer,
)
from .utils import check_adn_ordering, check_norm_layer


class MLP(BaseMLP):
    """Simple fully-connected neural network (or Multi-Layer Perceptron) with linear, normalization, activation
    and dropout layers.

    Works with 2D data (including batch dimension).

    Parameters
    ----------
    num_inputs : int
        Number of input features.
    num_outputs : int
        Number of outputs.
    hidden_dims : Sequence[int]
        Number of outputs for each hidden layer. Thus, this parameter also controls the number of hidden layers
        (equal to the length of the sequence).
    act : Optional[ActivationParameters], default="prelu"
        The activation function used after a linear layer, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions<nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network. Must be passed in the same way as ``act``.
        If ``None``, no last activation will be applied.
    norm : Optional[NormalizationParameters], default="batch"
        The normalization layer used after a linear layer, and optionally its arguments.
        Must be passed as ``norm_type`` or ``(norm_type, parameters)``. If ``None``, no normalization will be
        performed.\n
        ``norm_type`` can be any value in {``batch``, ``group``, ``instance``, ``layer``, ``syncbatch``}. Please refer to
        :torch:`PyTorch normalization layers <nn.html#normalization-layers>` to know the arguments for each of them.

        .. note::
            Please note that there's no need to pass the arguments ``num_channels``, ``num_features`` and ``normalized_shape``
            of the normalization layer, as they are automatically inferred from the output of the previous layer in the network.

    dropout : Optional[float], default=None
        Dropout ratio. If ``None``, no dropout.
    bias : bool, default=True
        Whether to have a bias term in linear layers.
    adn_ordering : str, default="NDA"
        Order of operations Activation, Dropout and Normalization, after a linear layer (except the last
        one). **Cannot contain duplicated letters**.
        For example if ``"ND"`` is passed, Normalization and then Dropout will be performed (without Activation).\n

        .. note::
            ADN will not be applied after the last linear layer.

    Raises
    ----------
    ValueError
        If the activation or normalization layer requires a mandatory argument, which is not passed by the user (via a dictionary
        in ``act`` or ``norm``).

    Examples
    --------

    .. code-block:: python

        >>> MLP(
                num_inputs=12,
                num_outputs=2,
                hidden_dims=[8, 4],
                dropout=0.1,
                act=("elu", {"alpha": 0.5}),
                norm=("group", {"num_groups": 2}),
                bias=True,
                adn_ordering="ADN",
                output_act="softmax",
            )
        MLP(
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (hidden0): Sequential(
                (linear): Linear(in_features=12, out_features=8, bias=True)
                (adn): ADN(
                    (A): ELU(alpha=0.5)
                    (D): Dropout(p=0.1, inplace=False)
                    (N): GroupNorm(2, 8, eps=1e-05, affine=True)
                )
            )
            (hidden1): Sequential(
                (linear): Linear(in_features=8, out_features=4, bias=True)
                (adn): ADN(
                    (A): ELU(alpha=0.5)
                    (D): Dropout(p=0.1, inplace=False)
                    (N): GroupNorm(2, 4, eps=1e-05, affine=True)
                )
            )
            (output): Sequential(
                (linear): Linear(in_features=4, out_features=2, bias=True)
                (output_act): Softmax(dim=None)
            )
        )
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_dims: Sequence[int],
        act: Optional[ActivationParameters] = ActFunction.PRELU,
        output_act: Optional[ActivationParameters] = None,
        norm: Optional[NormalizationParameters] = NormLayer.BATCH,
        dropout: Optional[float] = None,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        self.norm = check_norm_layer(norm)
        super().__init__(
            in_channels=num_inputs,
            out_channels=num_outputs,
            hidden_channels=hidden_dims,
            dropout=dropout,
            act=act,
            bias=bias,
            adn_ordering=check_adn_ordering(adn_ordering),
        )
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_dims = hidden_dims
        self.output = nn.Sequential(OrderedDict([("linear", self.output)]))
        self.output.output_act = get_act_layer(output_act) if output_act else None
        # renaming
        self._modules = OrderedDict(
            [
                (key.replace("hidden_", "hidden"), sub_m)
                for key, sub_m in self._modules.items()
            ]
        )

    def _get_layer(self, num_inputs: int, num_outputs: int, bias: bool) -> nn.Module:
        """
        Gets the parametrized Linear layer + ADN block.
        """
        if self.norm == NormLayer.LAYER:
            norm = ("layer", {"normalized_shape": num_outputs})
        else:
            norm = self.norm
        seq = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(num_inputs, num_outputs, bias)),
                    (
                        "adn",
                        ADN(
                            ordering=self.adn_ordering,
                            act=self.act,
                            norm=norm,
                            dropout=self.dropout,
                            dropout_dim=1,
                            in_channels=num_outputs,
                        ),
                    ),
                ]
            )
        )
        return seq
