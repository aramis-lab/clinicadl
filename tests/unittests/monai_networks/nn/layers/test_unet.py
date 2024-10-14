import numpy as np
import pytest
import torch

from clinicadl.monai_networks.nn.layers.unet import ConvBlock, DownBlock, UpSample

INPUT_1D = torch.randn(1, 1, 128)
INPUT_2D = torch.randn(1, 2, 32, 64)
INPUT_3D = torch.randn(1, 3, 32, 64, 16)


@pytest.mark.parametrize(
    "x,out_channels,act,dropout",
    [
        (INPUT_1D, 1, "sigmoid", 0.1),
        (INPUT_2D, 2, "elu", 0.0),
        (INPUT_3D, 5, ("elu", {"alpha": 0.1}), None),
    ],
)
def test_conv_block(x, out_channels, act, dropout):
    conv = ConvBlock(
        spatial_dims=len(x.shape) - 2,
        in_channels=x.shape[1],
        out_channels=out_channels,
        dropout=dropout,
        act=act,
    )
    out = conv(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == out_channels
    assert out.shape[2:] == x.shape[2:]


@pytest.mark.parametrize(
    "x,out_channels,act,dropout",
    [
        (INPUT_1D, 1, "sigmoid", 0.1),
        (INPUT_2D, 2, "elu", 0.0),
        (INPUT_3D, 5, ("elu", {"alpha": 0.1}), None),
    ],
)
def test_upsample(x, out_channels, act, dropout):
    up = UpSample(
        spatial_dims=len(x.shape) - 2,
        in_channels=x.shape[1],
        out_channels=out_channels,
        dropout=dropout,
        act=act,
    )
    out = up(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == out_channels
    assert (out.shape[2:] == np.array(x.shape[2:]) * 2).all()


@pytest.mark.parametrize(
    "x,out_channels,act,dropout",
    [
        (INPUT_1D, 1, "sigmoid", 0.1),
        (INPUT_2D, 2, "elu", 0.0),
        (INPUT_3D, 5, ("elu", {"alpha": 0.1}), None),
    ],
)
def test_down_block(x, out_channels, act, dropout):
    down = DownBlock(
        spatial_dims=len(x.shape) - 2,
        in_channels=x.shape[1],
        out_channels=out_channels,
        dropout=dropout,
        act=act,
    )
    out = down(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == out_channels
    assert (out.shape[2:] == np.array(x.shape[2:]) / 2).all()
