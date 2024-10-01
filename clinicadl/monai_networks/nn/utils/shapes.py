from math import ceil
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from clinicadl.monai_networks.nn.layers import PoolingLayer, UnpoolingLayer

__all__ = [
    "calculate_conv_out_shape",
    "calculate_convtranspose_out_shape",
    "calculate_pool_out_shape",
    "calculate_unpool_out_shape",
]


def calculate_conv_out_shape(
    in_shape: Union[Sequence[int], int],
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
    dilation: Union[Sequence[int], int],
) -> Tuple[int, ...]:
    """
    Calculates the output shape of a convolution layer. All arguments can be scalars or multiple
    values. Always return a tuple.
    """
    in_shape_np = np.atleast_1d(in_shape)
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    dilation_np = np.atleast_1d(dilation)

    out_shape_np = (
        (in_shape_np + 2 * padding_np - dilation_np * (kernel_size_np - 1) - 1)
        / stride_np
    ) + 1

    return tuple(int(s) for s in out_shape_np)


def calculate_convtranspose_out_shape(
    in_shape: Union[Sequence[int], int],
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
    output_padding: Union[Sequence[int], int],
    dilation: Union[Sequence[int], int],
    **kwargs,  # to pass arguments with layer.__dict__
) -> Tuple[int, ...]:
    """
    Calculates the output shape of a transposed convolution layer. All arguments can be scalars or
    multiple values. Always return a tuple.
    """
    in_shape_np = np.atleast_1d(in_shape)
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    dilation_np = np.atleast_1d(dilation)
    output_padding_np = np.atleast_1d(output_padding)

    out_shape_np = (
        (in_shape_np - 1) * stride_np
        - 2 * padding_np
        + dilation_np * (kernel_size_np - 1)
        + output_padding_np
        + 1
    )

    return tuple(int(s) for s in out_shape_np)


def calculate_pool_out_shape(
    pool_mode: Union[str, PoolingLayer],
    in_shape: Union[Sequence[int], int],
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
    dilation: Union[Sequence[int], int] = 1,
    ceil_mode: bool = False,
    **kwargs,  # to pass arguments with layer.__dict__
) -> Tuple[int, ...]:
    """
    Calculates the output shape of a pooling layer. The first argument is the type of pooling
    performed (`max` or `avg`). All other arguments can be scalars or multiple values, except
    `ceil_mode`.
    Always return a tuple.
    """
    pool_mode = PoolingLayer(pool_mode)
    if pool_mode == PoolingLayer.MAX:
        return _calculate_maxpool_out_shape(
            in_shape, kernel_size, stride, padding, dilation, ceil_mode
        )
    elif pool_mode == PoolingLayer.AVG:
        return _calculate_avgpool_out_shape(
            in_shape, kernel_size, stride, padding, ceil_mode
        )


def calculate_unpool_out_shape(
    unpool_mode: Union[str, UnpoolingLayer],
    in_shape: Union[Sequence[int], int],
    **kwargs,
) -> Tuple[int, ...]:
    """
    Calculates the output shape of an unpooling layer. The first argument is the type of unpooling
    performed (`upsample` or `convtranspose`).
    Always return a tuple.
    """
    unpool_mode = UnpoolingLayer(unpool_mode)
    if unpool_mode == UnpoolingLayer.UPSAMPLE:
        return _calculate_upsample_out_shape(in_shape, **kwargs)
    elif unpool_mode == UnpoolingLayer.CONV_TRANS:
        return calculate_convtranspose_out_shape(in_shape, **kwargs)


def _calculate_maxpool_out_shape(
    in_shape: Union[Sequence[int], int],
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
    dilation: Union[Sequence[int], int],
    ceil_mode: bool = False,
) -> Tuple[int, ...]:
    """
    Calculates the output shape of a MaxPool layer.
    """
    in_shape_np = np.atleast_1d(in_shape)
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    dilation_np = np.atleast_1d(dilation)

    out_shape_np = (
        (in_shape_np + 2 * padding_np - dilation_np * (kernel_size_np - 1) - 1)
        / stride_np
    ) + 1
    if ceil_mode:
        out_shape = tuple(ceil(s) for s in out_shape_np)
    else:
        out_shape = tuple(int(s) for s in out_shape_np)

    return out_shape


def _calculate_avgpool_out_shape(
    in_shape: Union[Sequence[int], int],
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
    ceil_mode: bool = False,
) -> Tuple[int, ...]:
    """
    Calculates the output shape of an AvgPool layer.
    """
    in_shape_np = np.atleast_1d(in_shape)
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_shape_np = ((in_shape_np + 2 * padding_np - kernel_size_np) / stride_np) + 1
    if ceil_mode:
        out_shape_np = np.ceil(out_shape_np)
        out_shape_np[(out_shape_np - 1) * stride_np >= in_shape_np + padding_np] -= 1

    return tuple(int(s) for s in out_shape_np)


def _calculate_upsample_out_shape(
    in_shape: Union[Sequence[int], int],
    scale_factor: Optional[int] = None,
    size: Optional[int] = None,
    **kwargs,
) -> Tuple[int, ...]:
    """
    Calculates the output shape of an Upsample layer.
    """
    if size and scale_factor:
        raise ValueError("Pass either size or scale_factor, not both.")
    elif size:
        return size
    elif scale_factor:
        in_shape_np = np.atleast_1d(in_shape)
        out_shape_np = in_shape_np * scale_factor
        return tuple(int(s) for s in out_shape_np)
    else:
        raise ValueError("Pass one of size or scale_factor.")
