from typing import Type, Union

import torch.nn as nn

from clinicadl.utils.enum import BaseEnum

from ..pool import PadMaxPool2d, PadMaxPool3d


class Pooling(str, BaseEnum):
    """Available pooling layers in ClinicaDL."""

    MAX = "MaxPool"
    PADMAX = "PadMaxPool"


class PoolLayer:
    """Factory object for creating Pooling layers."""

    def __new__(cls, pooling: Union[str, Pooling], dim: int) -> Type[nn.Module]:
        """
        Creates a Pooling layer.

        Parameters
        ----------
        pooling : Pooling
            Type of pooling.
        dim : int
            Dimension of the image.

        Returns
        -------
        Type[nn.Module]
            The normalization layer.
        """
        assert dim in {2, 3}, "Input dimension must be 2 or 3."
        pooling = Pooling(pooling)

        if pooling == Pooling.MAX:
            factory = _max_pool_factory
        elif pooling == Pooling.PADMAX:
            factory = _pad_max_pool_factory
        return factory(dim)


def _max_pool_factory(dim: int) -> Union[Type[nn.MaxPool2d], Type[nn.MaxPool3d]]:
    layers = (nn.MaxPool2d, nn.MaxPool3d)
    return layers[dim - 2]


def _pad_max_pool_factory(dim: int) -> Union[Type[PadMaxPool2d], Type[PadMaxPool3d]]:
    layers = (PadMaxPool2d, PadMaxPool3d)
    return layers[dim - 2]
