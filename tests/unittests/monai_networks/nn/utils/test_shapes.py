import pytest
import torch

from clinicadl.monai_networks.nn.utils.shapes import (
    _calculate_adaptivepool_out_shape,
    _calculate_avgpool_out_shape,
    _calculate_maxpool_out_shape,
    _calculate_upsample_out_shape,
    calculate_conv_out_shape,
    calculate_convtranspose_out_shape,
    calculate_pool_out_shape,
    calculate_unpool_out_shape,
)

INPUT_1D = torch.randn(2, 1, 10)
INPUT_2D = torch.randn(2, 1, 32, 32)
INPUT_3D = torch.randn(2, 1, 20, 21, 22)


@pytest.mark.parametrize(
    "input_tensor,kernel_size,stride,padding,dilation",
    [
        (INPUT_3D, 7, 2, (1, 2, 3), 3),
        (INPUT_2D, (5, 3), 1, 0, (2, 2)),
        (INPUT_1D, 3, 1, 2, 1),
    ],
)
def test_calculate_conv_out_shape(input_tensor, kernel_size, stride, padding, dilation):
    in_shape = input_tensor.shape[2:]
    dim = len(input_tensor.shape[2:])
    args = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
    }
    if dim == 1:
        conv = torch.nn.Conv1d
    elif dim == 2:
        conv = torch.nn.Conv2d
    else:
        conv = torch.nn.Conv3d

    output_shape = conv(**args)(input_tensor).shape[2:]
    assert (
        calculate_conv_out_shape(in_shape, kernel_size, stride, padding, dilation)
        == output_shape
    )


@pytest.mark.parametrize(
    "input_tensor,kernel_size,stride,padding,dilation,output_padding",
    [
        (INPUT_3D, 7, 2, (1, 2, 3), 3, 0),
        (INPUT_2D, (5, 3), 1, 0, (2, 2), (1, 0)),
        (INPUT_1D, 3, 3, 2, 1, 2),
    ],
)
def test_calculate_convtranspose_out_shape(
    input_tensor, kernel_size, stride, padding, dilation, output_padding
):
    in_shape = input_tensor.shape[2:]
    dim = len(input_tensor.shape[2:])
    args = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "output_padding": output_padding,
    }
    if dim == 1:
        conv = torch.nn.ConvTranspose1d
    elif dim == 2:
        conv = torch.nn.ConvTranspose2d
    else:
        conv = torch.nn.ConvTranspose3d

    output_shape = conv(**args)(input_tensor).shape[2:]
    assert (
        calculate_convtranspose_out_shape(
            in_shape, kernel_size, stride, padding, output_padding, dilation
        )
        == output_shape
    )


@pytest.mark.parametrize(
    "input_tensor,kernel_size,stride,padding,dilation,ceil_mode",
    [
        (INPUT_3D, 7, 2, (1, 2, 3), 3, False),
        (INPUT_3D, 7, 2, (1, 2, 3), 3, True),
        (INPUT_2D, (5, 3), 1, 0, (2, 2), False),
        (INPUT_2D, (5, 3), 1, 0, (2, 2), True),
        (INPUT_1D, 2, 1, 1, 1, False),
        (INPUT_1D, 2, 1, 1, 1, True),
    ],
)
def test_calculate_maxpool_out_shape(
    input_tensor, kernel_size, stride, padding, dilation, ceil_mode
):
    in_shape = input_tensor.shape[2:]
    dim = len(input_tensor.shape[2:])
    args = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
    }
    if dim == 1:
        max_pool = torch.nn.MaxPool1d
    elif dim == 2:
        max_pool = torch.nn.MaxPool2d
    else:
        max_pool = torch.nn.MaxPool3d

    output_shape = max_pool(**args)(input_tensor).shape[2:]
    assert (
        _calculate_maxpool_out_shape(
            in_shape, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode
        )
        == output_shape
    )


@pytest.mark.parametrize(
    "input_tensor,kernel_size,stride,padding,ceil_mode",
    [
        (INPUT_3D, 7, 2, (1, 2, 3), False),
        (INPUT_3D, 7, 2, (1, 2, 3), True),
        (INPUT_2D, (5, 3), 1, 0, False),
        (INPUT_2D, (5, 3), 1, 0, True),
        (INPUT_1D, 2, 1, 1, False),
        (INPUT_1D, 2, 1, 1, True),
        (
            INPUT_1D,
            2,
            3,
            1,
            True,
        ),  # special case with ceil_mode (see: https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html)
    ],
)
def test_calculate_avgpool_out_shape(
    input_tensor, kernel_size, stride, padding, ceil_mode
):
    in_shape = input_tensor.shape[2:]
    dim = len(in_shape)
    args = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
    }
    if dim == 1:
        avg_pool = torch.nn.AvgPool1d
    elif dim == 2:
        avg_pool = torch.nn.AvgPool2d
    else:
        avg_pool = torch.nn.AvgPool3d
    output_shape = avg_pool(**args)(input_tensor).shape[2:]
    assert (
        _calculate_avgpool_out_shape(
            in_shape, kernel_size, stride, padding, ceil_mode=ceil_mode
        )
        == output_shape
    )


@pytest.mark.parametrize(
    "input_tensor,kwargs",
    [
        (INPUT_3D, {"output_size": 1}),
        (INPUT_2D, {"output_size": (1, 2)}),
        (INPUT_1D, {"output_size": 3}),
    ],
)
def test_calculate_adaptivepool_out_shape(input_tensor, kwargs):
    in_shape = input_tensor.shape[2:]
    dim = len(in_shape)
    if dim == 1:
        avg_pool = torch.nn.AdaptiveAvgPool1d
        max_pool = torch.nn.AdaptiveMaxPool1d
    elif dim == 2:
        avg_pool = torch.nn.AdaptiveAvgPool2d
        max_pool = torch.nn.AdaptiveMaxPool2d
    else:
        avg_pool = torch.nn.AdaptiveAvgPool3d
        max_pool = torch.nn.AdaptiveMaxPool3d

    output_shape = max_pool(**kwargs)(input_tensor).shape[2:]
    assert _calculate_adaptivepool_out_shape(in_shape, **kwargs) == output_shape

    output_shape = avg_pool(**kwargs)(input_tensor).shape[2:]
    assert _calculate_adaptivepool_out_shape(in_shape, **kwargs) == output_shape


def test_calculate_pool_out_shape():
    in_shape = INPUT_3D.shape[2:]
    assert calculate_pool_out_shape(
        pool_mode="max",
        in_shape=in_shape,
        kernel_size=7,
        stride=2,
        padding=(1, 2, 3),
        dilation=3,
        ceil_mode=True,
    ) == (3, 4, 6)
    assert calculate_pool_out_shape(
        pool_mode="avg",
        in_shape=in_shape,
        kernel_size=7,
        stride=2,
        padding=(1, 2, 3),
        ceil_mode=True,
    ) == (9, 10, 12)
    assert calculate_pool_out_shape(
        pool_mode="adaptiveavg",
        in_shape=in_shape,
        output_size=(3, 4, 5),
    ) == (3, 4, 5)
    assert calculate_pool_out_shape(
        pool_mode="adaptivemax",
        in_shape=in_shape,
        output_size=1,
    ) == (1, 1, 1)
    with pytest.raises(ValueError):
        calculate_pool_out_shape(
            pool_mode="abc",
            in_shape=in_shape,
            kernel_size=7,
            stride=2,
            padding=(1, 2, 3),
            dilation=3,
            ceil_mode=True,
        )


@pytest.mark.parametrize(
    "input_tensor,kwargs",
    [
        (INPUT_3D, {"scale_factor": 2}),
        (INPUT_2D, {"size": (40, 41)}),
        (INPUT_2D, {"size": 40}),
        (INPUT_2D, {"scale_factor": (3, 2)}),
        (INPUT_1D, {"scale_factor": 2}),
    ],
)
def test_calculate_upsample_out_shape(input_tensor, kwargs):
    in_shape = input_tensor.shape[2:]
    unpool = torch.nn.Upsample(**kwargs)

    output_shape = unpool(input_tensor).shape[2:]
    assert _calculate_upsample_out_shape(in_shape, **kwargs) == output_shape


def test_calculate_unpool_out_shape():
    in_shape = INPUT_3D.shape[2:]
    assert calculate_unpool_out_shape(
        unpool_mode="convtranspose",
        in_shape=in_shape,
        kernel_size=5,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
    ) == (24, 25, 26)
    assert calculate_unpool_out_shape(
        unpool_mode="upsample",
        in_shape=in_shape,
        scale_factor=2,
    ) == (40, 42, 44)
    with pytest.raises(ValueError):
        calculate_unpool_out_shape(
            unpool_mode="abc",
            in_shape=in_shape,
        )
