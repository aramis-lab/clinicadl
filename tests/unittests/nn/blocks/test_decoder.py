import pytest
import torch

import clinicadl.nn.blocks.decoder as decoder


@pytest.fixture
def input_2d():
    return torch.randn(2, 1, 10, 10)


@pytest.fixture
def input_3d():
    return torch.randn(2, 1, 10, 10, 10)


@pytest.fixture
def latent_vector():
    return torch.randn(2, 3)


@pytest.fixture(params=["latent_vector", "input_2d"])
def to_decode(request):
    return request.getfixturevalue(request.param)


def test_decoder2d(input_2d):
    network = decoder.Decoder2D(
        input_channels=input_2d.shape[1], output_channels=(input_2d.shape[1] + 3)
    )
    output_2d = network(input_2d)
    assert output_2d.shape[1] == input_2d.shape[1] + 3
    assert len(output_2d.shape) == 4


def test_vae_decoder2d(to_decode):
    latent_dim = 1 if len(to_decode.shape) == 2 else 2

    network = decoder.VAE_Decoder2D(
        input_shape=(1, 5, 5),
        latent_size=to_decode.shape[1],
        n_conv=1,
        last_layer_channels=2,
        latent_dim=latent_dim,
        feature_size=4,
    )
    output_2d = network(to_decode)
    assert len(output_2d.shape) == 4
    assert output_2d.shape[0] == 2
    assert output_2d.shape[1] == 1


def test_decoder3d(input_3d):
    network = decoder.Decoder3D(
        input_channels=input_3d.shape[1], output_channels=(input_3d.shape[1] + 3)
    )
    output_3d = network(input_3d)
    assert output_3d.shape[1] == input_3d.shape[1] + 3
    assert len(output_3d.shape) == 5
