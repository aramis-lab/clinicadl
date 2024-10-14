import pytest
import torch
from torch.nn import Flatten, Linear, Softmax

from clinicadl.monai_networks.nn import CNN, MLP, ConvEncoder

INPUT_1D = torch.randn(3, 1, 16)
INPUT_2D = torch.randn(3, 1, 15, 16)
INPUT_3D = torch.randn(3, 3, 20, 21, 22)


@pytest.mark.parametrize("input_tensor", [INPUT_1D, INPUT_2D, INPUT_3D])
@pytest.mark.parametrize("channels", [(), (2, 4)])
@pytest.mark.parametrize(
    "mlp_args", [None, {"hidden_channels": []}, {"hidden_channels": (2, 4)}]
)
def test_cnn(input_tensor, channels, mlp_args):
    in_shape = input_tensor.shape[1:]
    net = CNN(
        in_shape=in_shape,
        num_outputs=2,
        conv_args={"channels": channels},
        mlp_args=mlp_args,
    )
    output = net(input_tensor)
    assert output.shape == (3, 2)
    assert isinstance(net.convolutions, ConvEncoder)
    assert isinstance(net.mlp, MLP)

    if mlp_args is None or mlp_args["hidden_channels"] == []:
        children = net.mlp.children()
        assert isinstance(next(children), Flatten)
        assert isinstance(next(children), Linear)
        with pytest.raises(StopIteration):
            next(children)

    if channels == []:
        with pytest.raises(StopIteration):
            next(net.convolutions.parameters())


@pytest.mark.parametrize(
    "conv_args,mlp_args",
    [
        (None, {"hidden_channels": [2]}),
        ({"channels": [2]}, {}),
    ],
)
def test_checks(conv_args, mlp_args):
    with pytest.raises(ValueError):
        CNN(in_shape=(1, 10, 10), num_outputs=2, conv_args=conv_args, mlp_args=mlp_args)


def test_params():
    conv_args = {"channels": [2], "act": "celu"}
    mlp_args = {"hidden_channels": [2], "act": "relu", "output_act": "softmax"}
    net = CNN(
        in_shape=(1, 10, 10), num_outputs=2, conv_args=conv_args, mlp_args=mlp_args
    )
    assert net.convolutions.act == "celu"
    assert net.mlp.act == "relu"
    assert isinstance(net.mlp.output.output_act, Softmax)
