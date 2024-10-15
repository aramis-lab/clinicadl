import pytest
import torch
from torch.nn import GELU, Sigmoid, Tanh

from clinicadl.monai_networks.nn import AutoEncoder
from clinicadl.monai_networks.nn.layers.utils import ActFunction


@pytest.mark.parametrize(
    "input_tensor,kernel_size,stride,padding,dilation,pooling,pooling_indices",
    [
        (torch.randn(2, 1, 21), 3, 1, 0, 1, ("max", {"kernel_size": 2}), [0]),
        (
            torch.randn(2, 1, 65, 85),
            (3, 5),
            (1, 2),
            0,
            (1, 2),
            ("max", {"kernel_size": 2, "stride": 1}),
            [0],
        ),
        (
            torch.randn(2, 1, 64, 62, 61),  # to test output padding
            4,
            2,
            (1, 1, 0),
            1,
            ("avg", {"kernel_size": 3, "stride": 2}),
            [-1],
        ),
        (
            torch.randn(2, 1, 51, 55, 45),
            3,
            2,
            0,
            1,
            ("max", {"kernel_size": 2, "ceil_mode": True}),
            [0, 1, 2],
        ),
        (
            torch.randn(2, 1, 51, 55, 45),
            3,
            2,
            0,
            1,
            [
                ("max", {"kernel_size": 2, "ceil_mode": True}),
                ("max", {"kernel_size": 2, "stride": 1, "ceil_mode": False}),
            ],
            [0, 1],
        ),
    ],
)
def test_output_shape(
    input_tensor, kernel_size, stride, padding, dilation, pooling, pooling_indices
):
    net = AutoEncoder(
        in_shape=input_tensor.shape[1:],
        latent_size=3,
        conv_args={
            "channels": [2, 4, 8],
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "pooling": pooling,
            "pooling_indices": pooling_indices,
        },
    )
    output = net(input_tensor)
    assert output.shape == input_tensor.shape


def test_out_channels():
    input_tensor = torch.randn(2, 1, 64, 62, 61)
    net = AutoEncoder(
        in_shape=input_tensor.shape[1:],
        latent_size=3,
        conv_args={"channels": [2, 4, 8]},
        mlp_args={"hidden_channels": [8, 4]},
        out_channels=3,
    )
    assert net(input_tensor).shape == (2, 3, 64, 62, 61)
    assert net.decoder.convolutions.layer2.conv.in_channels == 2
    assert net.decoder.convolutions.layer2.conv.out_channels == 3


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_out_activation(act):
    input_tensor = torch.randn(2, 1, 32, 32)
    net = AutoEncoder(
        in_shape=(1, 32, 32),
        latent_size=3,
        conv_args={"channels": [2]},
        output_act=act,
    )
    assert net(input_tensor).shape == (2, 1, 32, 32)


def test_params():
    net = AutoEncoder(
        in_shape=(1, 100, 100),
        latent_size=3,
        conv_args={"channels": [2], "act": "celu", "output_act": "sigmoid"},
        mlp_args={"hidden_channels": [2], "act": "relu", "output_act": "gelu"},
        output_act="tanh",
        out_channels=2,
    )
    assert net.encoder.convolutions.act == "celu"
    assert net.decoder.convolutions.act == "celu"
    assert net.encoder.mlp.act == "relu"
    assert net.decoder.mlp.act == "relu"
    assert isinstance(net.encoder.mlp.output.output_act, GELU)
    assert isinstance(net.encoder.mlp.output.output_act, GELU)
    assert isinstance(net.encoder.convolutions.output_act, Sigmoid)
    assert isinstance(net.decoder.convolutions.output_act, Tanh)


@pytest.mark.parametrize(
    "in_shape,upsampling_mode,error",
    [
        ((1, 10), "bilinear", True),
        ((1, 10, 10), "linear", True),
        ((1, 10, 10), "trilinear", True),
        ((1, 10, 10, 10), "bicubic", True),
        ((1, 10), "linear", False),
        ((1, 10, 10), "bilinear", False),
        ((1, 10, 10, 10), "trilinear", False),
    ],
)
def test_checks(in_shape, upsampling_mode, error):
    if error:
        with pytest.raises(ValueError):
            AutoEncoder(
                in_shape=in_shape,
                latent_size=3,
                conv_args={"channels": []},
                upsampling_mode=upsampling_mode,
            )
    else:
        AutoEncoder(
            in_shape=in_shape,
            latent_size=3,
            conv_args={"channels": []},
            upsampling_mode=upsampling_mode,
        )
