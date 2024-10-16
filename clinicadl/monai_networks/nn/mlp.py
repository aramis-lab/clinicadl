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
from .utils import check_norm_layer


class MLP(BaseMLP):
    """Simple full-connected layer neural network (or Multi-Layer Perceptron) with linear, normalization, activation
    and dropout layers.

    Parameters
    ----------
    in_channels : int
        number of input channels (i.e. number of features).
    out_channels : int
        number of output channels.
    hidden_channels : Sequence[int]
        number of output channels for each hidden layer. Thus, this parameter also controls the number of hidden layers.
    act : Optional[ActivationParameters] (optional, default=ActFunction.PRELU)
        the activation function used after a linear layer, and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.
    output_act : Optional[ActivationParameters] (optional, default=None)
        a potential activation layer applied to the output of the network. Should be pass in the same way as `act`.
        If None, no last activation will be applied.
    norm : Optional[NormalizationParameters] (optional, default=NormLayer.BATCH)
        the normalization type used after a linear layer, and optionally the arguments of the normalization
        layer. Should be passed as `norm_type` or `(norm_type, parameters)`. If None, no normalization will be
        performed.\n
        `norm_type` can be any value in {`batch`, `group`, `instance`, `layer`, `syncbatch`}. Please refer to PyTorch's
        [normalization layers](https://pytorch.org/docs/stable/nn.html#normalization-layers) to know the mandatory and
        optional arguments for each of them.\n
        Please note that arguments `num_channels`, `num_features` and `normalized_shape` of the normalization layer
        should not be passed, as they are automatically inferred from the output of the previous layer in the network.
    dropout : Optional[float] (optional, default=None)
        dropout ratio. If None, no dropout.
    bias : bool (optional, default=True)
        whether to have a bias term in linear layers.
    adn_ordering : str (optional, default="NDA")
        order of operations `Activation`, `Dropout` and `Normalization` after a linear layer (except the last
        one).
        For example if "ND" is passed, `Normalization` and then `Dropout` will be performed (without `Activation`).\n
        Note: ADN will not be applied after the last linear layer.

    Examples
    --------
    >>> MLP(in_channels=12, out_channels=2, hidden_channels=[8, 4], dropout=0.1, act=("elu", {"alpha": 0.5}),
        norm=("group", {"num_groups": 2}), bias=True, adn_ordering="ADN", output_act="softmax")
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
        in_channels: int,
        out_channels: int,
        hidden_channels: Sequence[int],
        act: Optional[ActivationParameters] = ActFunction.PRELU,
        output_act: Optional[ActivationParameters] = None,
        norm: Optional[NormalizationParameters] = NormLayer.BATCH,
        dropout: Optional[float] = None,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        self.norm = check_norm_layer(norm)
        super().__init__(
            in_channels, out_channels, hidden_channels, dropout, act, bias, adn_ordering
        )
        self.output = nn.Sequential(OrderedDict([("linear", self.output)]))
        self.output.output_act = get_act_layer(output_act) if output_act else None
        # renaming
        self._modules = OrderedDict(
            [
                (key.replace("hidden_", "hidden"), sub_m)
                for key, sub_m in self._modules.items()
            ]
        )

    def _get_layer(self, in_channels: int, out_channels: int, bias: bool) -> nn.Module:
        """
        Gets the parametrized Linear layer + ADN block.
        """
        if self.norm == NormLayer.LAYER:
            norm = ("layer", {"normalized_shape": out_channels})
        else:
            norm = self.norm
        seq = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(in_channels, out_channels, bias)),
                    (
                        "adn",
                        ADN(
                            ordering=self.adn_ordering,
                            act=self.act,
                            norm=norm,
                            dropout=self.dropout,
                            dropout_dim=1,
                            in_channels=out_channels,
                        ),
                    ),
                ]
            )
        )
        return seq
