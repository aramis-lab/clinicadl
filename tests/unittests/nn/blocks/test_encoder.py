import pytest
import torch

import clinicadl.nn.blocks.encoder as encoder


@pytest.fixture
def input_2d():
    return torch.randn(2, 1, 10, 10)


@pytest.fixture
def input_3d():
    return torch.randn(2, 1, 10, 10, 10)


def test_encoder2d(input_2d):
    network = encoder.Encoder2D(
        input_channels=input_2d.shape[1], output_channels=(input_2d.shape[1] + 3)
    )
    output_2d = network(input_2d)
    assert output_2d.shape[1] == input_2d.shape[1] + 3
    assert len(output_2d.shape) == 4


@pytest.mark.parametrize("latent_dim", [1, 2])
def test_vae_encoder2d(latent_dim, input_2d):
    network = encoder.VAE_Encoder2D(
        input_shape=(1, 10, 10),
        n_conv=1,
        first_layer_channels=4,
        latent_dim=latent_dim,
        feature_size=4,
    )
    output = network(input_2d)
    assert output.shape[0] == 2
    assert len(output.shape) == 2 if latent_dim == 1 else 4


def test_encoder3d(input_3d):
    network = encoder.Encoder3D(
        input_channels=input_3d.shape[1], output_channels=(input_3d.shape[1] + 3)
    )
    output_3d = network(input_3d)
    assert output_3d.shape[1] == input_3d.shape[1] + 3
    assert len(output_3d.shape) == 5
