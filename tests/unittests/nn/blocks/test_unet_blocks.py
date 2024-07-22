import pytest
import torch


@pytest.fixture
def input_3d():
    return torch.randn(2, 4, 10, 10, 10)


@pytest.fixture
def skip_input():
    return torch.randn(2, 4, 10, 10, 10)


def test_UNetDown(input_3d, helpers):
    from clinicadl.nn.blocks import UNetDown

    layer = UNetDown(in_size=input_3d.shape[1], out_size=8)
    out = layer(input_3d)
    assert out.shape[:2] == torch.Size((input_3d.shape[0], 8))


def test_UNetUp(input_3d, skip_input, helpers):
    from clinicadl.nn.blocks import UNetUp

    layer = UNetUp(in_size=input_3d.shape[1] * 2, out_size=2)
    out = layer(input_3d, skip_input=skip_input)
    assert out.shape[:2] == torch.Size((input_3d.shape[0], 2))


def test_UNetFinalLayer(input_3d, skip_input, helpers):
    from clinicadl.nn.blocks import UNetFinalLayer

    layer = UNetFinalLayer(in_size=input_3d.shape[1] * 2, out_size=2)
    out = layer(input_3d, skip_input=skip_input)
    assert out.shape[:2] == torch.Size((input_3d.shape[0], 2))
