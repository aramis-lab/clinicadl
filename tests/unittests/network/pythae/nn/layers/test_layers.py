import pytest
import torch

import clinicadl.network.pythae.nn.layers as layers


@pytest.fixture
def input_2d():
    return torch.randn(2, 1, 5, 5)


@pytest.fixture
def input_3d():
    return torch.randn(2, 1, 5, 5, 5)


def test_pool_layers(input_2d, input_3d):  # TODO : test unpool
    output_3d = layers.PadMaxPool3d(kernel_size=2, stride=1)(input_3d)
    output_2d = layers.PadMaxPool2d(kernel_size=2, stride=1)(input_2d)

    assert len(output_3d.shape) == 5  # TODO : test more precisely
    assert output_3d.shape[0] == 2
    assert len(output_2d.shape) == 4
    assert output_2d.shape[0] == 2


def test_flatten_layers(input_2d, input_3d):
    output_3d = layers.Flatten()(input_3d)
    output_2d = layers.Flatten()(input_2d)

    assert output_3d.shape == torch.Size((2, 1 * 5 * 5 * 5))
    assert output_2d.shape == torch.Size((2, 1 * 5 * 5))


def test_unflatten_layers():
    flattened_2d = torch.randn(2, 1 * 5 * 4)
    flattened_3d = torch.randn(2, 1 * 5 * 4 * 3)

    output_3d = layers.Unflatten3D(channel=1, height=5, width=4, depth=3)(flattened_3d)
    output_2d = layers.Unflatten2D(channel=1, height=5, width=4)(flattened_2d)

    assert output_3d.shape == torch.Size((2, 1, 5, 4, 3))
    assert output_2d.shape == torch.Size((2, 1, 5, 4))
