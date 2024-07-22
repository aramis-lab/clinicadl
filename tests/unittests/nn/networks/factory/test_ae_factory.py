import pytest
import torch
import torch.nn as nn

from clinicadl.nn.layers import (
    PadMaxPool2d,
    PadMaxPool3d,
)


@pytest.fixture
def input_3d():
    return torch.randn(2, 4, 10, 10, 10)


@pytest.fixture
def input_2d():
    return torch.randn(2, 4, 10, 10)


@pytest.fixture
def cnn3d():
    class CNN(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.convolutions = nn.Sequential(
                nn.Conv3d(in_channels=input_size[0], out_channels=4, kernel_size=3),
                nn.BatchNorm3d(num_features=4),
                nn.LeakyReLU(),
                PadMaxPool3d(kernel_size=2, stride=1, return_indices=False),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(42, 2),
            )

    return CNN


@pytest.fixture
def cnn2d():
    class CNN(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.convolutions = nn.Sequential(
                nn.Conv2d(in_channels=input_size[0], out_channels=4, kernel_size=3),
                nn.BatchNorm2d(num_features=4),
                nn.LeakyReLU(),
                PadMaxPool2d(kernel_size=2, stride=1, return_indices=False),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(42, 2),  # should not raise an error
            )

    return CNN


@pytest.mark.parametrize("input, cnn", [("input_3d", "cnn3d"), ("input_2d", "cnn2d")])
def test_autoencoder_from_cnn(input, cnn, request):
    from clinicadl.nn.networks.ae import AE
    from clinicadl.nn.networks.factory import autoencoder_from_cnn

    input_ = request.getfixturevalue(input)
    cnn = request.getfixturevalue(cnn)(input_size=input_.shape[1:])
    encoder, decoder = autoencoder_from_cnn(cnn)
    autoencoder = AE(encoder, decoder)
    assert autoencoder(input_).shape == input_.shape
