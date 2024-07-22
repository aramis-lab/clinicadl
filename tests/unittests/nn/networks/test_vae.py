import pytest
import torch

import clinicadl.nn.networks.vae as vae


@pytest.fixture
def input_2d():
    return torch.randn(2, 3, 100, 100)


@pytest.fixture
def input_3d():
    return torch.randn(2, 1, 50, 50, 50)


@pytest.mark.parametrize(
    "input_,network,latent_space_size",
    [
        (
            torch.randn(2, 3, 100, 100),
            vae.VanillaDenseVAE(
                input_size=(3, 100, 100), latent_space_size=10, feature_size=100
            ),
            10,
        ),
        (
            torch.randn(2, 1, 80, 96, 80),
            vae.VanillaDenseVAE3D(
                size_reduction_factor=2,
                latent_space_size=10,
                feature_size=100,
            ),
            10,
        ),
        # (
        #     torch.randn(2, 1, 50, 50, 50), # TODO : only work with certain size
        #     vae.CVAE_3D(
        #         input_size=(3, 50, 50, 50),
        #         latent_space_size=10,
        #     ),
        #     10,
        # ),
        (
            torch.randn(2, 1, 56, 64, 56),
            vae.CVAE_3D_final_conv(
                size_reduction_factor=3,
                latent_space_size=10,
            ),
            10,
        ),
        (
            torch.randn(2, 1, 32, 40, 32),
            vae.CVAE_3D_half(
                size_reduction_factor=5,
                latent_space_size=10,
            ),
            10,
        ),
    ],
)
def test_DenseVAEs(input_, network, latent_space_size):
    output = network(input_)

    assert output[0].shape == torch.Size((input_.shape[0], latent_space_size))
    assert output[1].shape == torch.Size((input_.shape[0], latent_space_size))
    assert output[2].shape == input_.shape


@pytest.mark.parametrize(
    "input_,network",
    [
        (
            torch.randn(2, 3, 100, 100),
            vae.VanillaSpatialVAE(input_size=(3, 100, 100)),
        ),
        # (torch.randn(2, 3, 100, 100, 100), vae.VanillaSpatialVAE3D(input_size=(3, 100, 100, 100))),   # TODO : output doesn't have the same size
    ],
)
def test_SpatialVAEs(input_, network):
    output = network(input_)

    assert output[0].shape[:2] == torch.Size((input_.shape[0], 1))
    assert len(output[0].shape) == len(input_.shape)
    assert output[1].shape[:2] == torch.Size((input_.shape[0], 1))
    assert len(output[0].shape) == len(input_.shape)
    assert output[2].shape == input_.shape
