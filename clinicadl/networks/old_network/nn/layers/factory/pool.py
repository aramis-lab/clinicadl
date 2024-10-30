from typing import TYPE_CHECKING, Type, Union

import torch.nn as nn

from clinicadl.utils.enum import BaseEnum

from ..pool import PadMaxPool2d, PadMaxPool3d


class Pooling(str, BaseEnum):
    """Available pooling layers in ClinicaDL."""

    MAX = "MaxPool"
    PADMAX = "PadMaxPool"


def get_pool_layer(pooling: Union[str, Pooling], dim: int) -> Type[nn.Module]:
    """
    A factory object for creating Pooling layers.

    Parameters
    ----------
    pooling : Pooling
        Type of pooling.
    dim : int
        Dimension of the image.

    Returns
    -------
    Type[nn.Module]
        The pooling layer.

    Raises
    ------
    AssertionError
        If dim is not 2 or 3.
    """
    assert dim in {2, 3}, "Input dimension must be 2 or 3."
    pooling = Pooling(pooling)

    if pooling == Pooling.MAX:
        factory = _max_pool_factory
    elif pooling == Pooling.PADMAX:
        factory = _pad_max_pool_factory
    return factory(dim)


def _max_pool_factory(dim: int) -> Union[Type[nn.MaxPool2d], Type[nn.MaxPool3d]]:
    """
    A factory object for creating Max Pooling layers.

    Parameters
    ----------
    dim : int
        Dimension of the image.

    Returns
    -------
    Union[Type[nn.MaxPool2d], Type[nn.MaxPool3d]]
        The pooling layer.
    """
    layers = (nn.MaxPool2d, nn.MaxPool3d)
    return layers[dim - 2]


def _pad_max_pool_factory(dim: int) -> Union[Type[PadMaxPool2d], Type[PadMaxPool3d]]:
    """
    A factory object for creating Pad-Max Pooling layers.

    Parameters
    ----------
    dim : int
        Dimension of the image.

    Returns
    -------
    Union[Type[PadMaxPool2d], Type[PadMaxPool3d]]
        The pooling layer.
    """
    layers = (PadMaxPool2d, PadMaxPool3d)
    return layers[dim - 2]
