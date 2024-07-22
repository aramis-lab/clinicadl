import pytest
import torch

import clinicadl.nn.layers as layers


@pytest.fixture
def input_2d():
    return torch.randn(2, 1, 5, 5)


@pytest.fixture
def input_3d():
    return torch.randn(2, 1, 5, 5, 5)


def test_pool_layers(input_2d, input_3d):
    output_3d = layers.PadMaxPool3d(kernel_size=2, stride=1)(input_3d)
    output_2d = layers.PadMaxPool2d(kernel_size=2, stride=1)(input_2d)

    assert len(output_3d.shape) == 5  # TODO : test more precisely and test padding
    assert output_3d.shape[0] == 2
    assert len(output_2d.shape) == 4
    assert output_2d.shape[0] == 2


def test_unpool_layers():  # TODO : test padding
    import torch.nn as nn

    pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    unpool = layers.CropMaxUnpool2d(2, stride=2)
    input_ = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        ]
    )
    excpected_output = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0, 8.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 14.0, 0.0, 16.0],
                ]
            ]
        ]
    )
    output, indices = pool(input_)
    assert (unpool(output, indices) == excpected_output).all()

    pool = nn.MaxPool3d(2, stride=1, return_indices=True)
    unpool = layers.CropMaxUnpool3d(2, stride=1)
    input_ = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
    excpected_output = torch.tensor(
        [[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 8.0]]]]
    )
    output, indices = pool(input_)
    assert (unpool(output, indices) == excpected_output).all()


def test_unflatten_layers():
    flattened_2d = torch.randn(2, 1 * 5 * 4)
    flattened_3d = torch.randn(2, 1 * 5 * 4 * 3)

    output_3d = layers.Unflatten3D(channel=1, height=5, width=4, depth=3)(flattened_3d)
    output_2d = layers.Unflatten2D(channel=1, height=5, width=4)(flattened_2d)

    assert output_3d.shape == torch.Size((2, 1, 5, 4, 3))
    assert output_2d.shape == torch.Size((2, 1, 5, 4))


def test_reshape_layers(input_2d):
    reshape = layers.Reshape((2, 1, 25))
    assert reshape(input_2d).shape == torch.Size((2, 1, 25))


def test_gradient_reversal(input_3d):
    from copy import deepcopy

    import torch.nn as nn

    input_ = torch.randn(2, 5)
    ref_ = torch.randn(2, 3)
    layer = nn.Linear(5, 3)
    reversed_layer = nn.Sequential(deepcopy(layer), layers.GradientReversal(alpha=2.0))
    criterion = torch.nn.MSELoss()

    criterion(layer(input_), ref_).backward()
    criterion(reversed_layer(input_), ref_).backward()
    assert all(
        (p2.grad == -2.0 * p1.grad).all()
        for p1, p2 in zip(layer.parameters(), reversed_layer.parameters())
    )
