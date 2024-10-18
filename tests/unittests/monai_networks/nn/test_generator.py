import pytest
import torch
from torch.nn import Flatten, Linear

from clinicadl.monai_networks.nn import MLP, ConvDecoder, Generator


@pytest.fixture
def input_tensor():
    return torch.randn(2, 8)


@pytest.mark.parametrize("channels", [(), (2, 4)])
@pytest.mark.parametrize(
    "mlp_args", [None, {"hidden_channels": []}, {"hidden_channels": (2, 4)}]
)
@pytest.mark.parametrize("start_shape", [(1, 5), (1, 5, 5), (1, 5, 5)])
def test_generator(input_tensor, start_shape, channels, mlp_args):
    latent_size = input_tensor.shape[1]
    net = Generator(
        latent_size=latent_size,
        start_shape=start_shape,
        conv_args={"channels": channels},
        mlp_args=mlp_args,
    )
    output = net(input_tensor)
    assert output.shape[1:] == net.output_shape
    assert isinstance(net.convolutions, ConvDecoder)
    assert isinstance(net.mlp, MLP)

    if mlp_args is None or mlp_args["hidden_channels"] == []:
        children = net.mlp.children()
        assert isinstance(next(children), Flatten)
        assert isinstance(next(children).linear, Linear)
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
        Generator(
            latent_size=2,
            start_shape=(1, 10, 10),
            conv_args=conv_args,
            mlp_args=mlp_args,
        )


def test_params():
    conv_args = {"channels": [2], "act": "celu"}
    mlp_args = {"hidden_channels": [2], "act": "relu"}
    net = Generator(
        latent_size=2, start_shape=(1, 10, 10), conv_args=conv_args, mlp_args=mlp_args
    )
    assert net.convolutions.act == "celu"
    assert net.mlp.act == "relu"
