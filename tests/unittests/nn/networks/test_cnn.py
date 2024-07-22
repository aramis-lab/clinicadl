import pytest
import torch

import clinicadl.nn.networks.cnn as cnn


@pytest.fixture
def input_2d():
    return torch.randn(2, 3, 100, 100)


@pytest.fixture
def input_3d():
    return torch.randn(2, 1, 100, 100, 100)


@pytest.mark.parametrize("network", [net.value for net in cnn.CNN2d])
def test_2d_cnn(network, input_2d):
    network = getattr(cnn, network)(
        input_size=input_2d.shape[1:], output_size=3, dropout=0.5
    )
    output_2d = network(input_2d)
    assert output_2d.shape == (2, 3)


@pytest.mark.parametrize("network", [net.value for net in cnn.CNN3d])
def test_3d_cnn(network, input_3d):
    network = getattr(cnn, network)(
        input_size=input_3d.shape[1:], output_size=1, dropout=0.5
    )
    output_2d = network(input_3d)
    assert output_2d.shape == (2, 1)
