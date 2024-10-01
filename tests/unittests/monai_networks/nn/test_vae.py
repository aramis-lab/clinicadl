import pytest
import torch
from numpy import isclose
from torch.nn import ReLU

from clinicadl.monai_networks.nn import VAE


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
            [0],
        ),
        (
            torch.randn(2, 1, 51, 55, 45),
            3,
            2,
            0,
            1,
            ("max", {"kernel_size": 2, "ceil_mode": True}),
            [0, 1],
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
    latent_size = 3
    net = VAE(
        in_shape=input_tensor.shape[1:],
        latent_size=latent_size,
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
    recon, mu, log_var = net(input_tensor)
    assert recon.shape == input_tensor.shape
    assert mu.shape == (input_tensor.shape[0], latent_size)
    assert log_var.shape == (input_tensor.shape[0], latent_size)


def test_mu_log_var():
    net = VAE(
        in_shape=(1, 5, 5),
        latent_size=4,
        conv_args={"channels": []},
        mlp_args={"hidden_channels": [12], "output_act": "relu", "act": "celu"},
    )
    assert net.mu.linear.in_features == 12
    assert net.log_var.linear.in_features == 12
    assert isinstance(net.mu.output_act, ReLU)
    assert isinstance(net.log_var.output_act, ReLU)
    assert net.encoder(torch.randn(2, 1, 5, 5)).shape == (2, 12)
    _, mu, log_var = net(torch.randn(2, 1, 5, 5))
    assert not isclose(mu.detach().numpy(), log_var.detach().numpy()).all()

    net = VAE(
        in_shape=(1, 5, 5),
        latent_size=4,
        conv_args={"channels": []},
        mlp_args={"hidden_channels": [12]},
    )
    assert net.mu.in_features == 12
    assert net.log_var.in_features == 12
