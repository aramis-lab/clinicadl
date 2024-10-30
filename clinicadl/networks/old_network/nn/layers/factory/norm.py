from typing import Type, Union

import torch.nn as nn

from clinicadl.utils.enum import BaseEnum


class Normalization(str, BaseEnum):  # TODO : remove from global enum
    """Available normalization layers in ClinicaDL."""

    BATCH = "BatchNorm"
    GROUP = "GroupNorm"
    INSTANCE = "InstanceNorm"


def get_norm_layer(
    normalization: Union[str, Normalization], dim: int
) -> Type[nn.Module]:
    """
    A factory function for creating Normalization layers.

    Parameters
    ----------
    normalization : Normalization
        Type of normalization.
    dim : int
        Dimension of the image.

    Returns
    -------
    Type[nn.Module]
        The normalization layer.

    Raises
    ------
    AssertionError
        If dim is not 2 or 3.
    """
    assert dim in {2, 3}, "Input dimension must be 2 or 3."
    normalization = Normalization(normalization)

    if normalization == Normalization.BATCH:
        factory = _batch_norm_factory
    elif normalization == Normalization.INSTANCE:
        factory = _instance_norm_factory
    elif normalization == Normalization.GROUP:
        factory = _group_norm_factory
    return factory(dim)


def _instance_norm_factory(
    dim: int,
) -> Union[Type[nn.InstanceNorm2d], Type[nn.InstanceNorm3d]]:
    """
    A factory function for creating Instance Normalization layers.

    Parameters
    ----------
    dim : int
        Dimension of the image.

    Returns
    -------
    Union[Type[nn.InstanceNorm2d], Type[nn.InstanceNorm3d]]
        The normalization layer.
    """
    layers = (nn.InstanceNorm2d, nn.InstanceNorm3d)
    return layers[dim - 2]


def _batch_norm_factory(dim: int) -> Union[Type[nn.BatchNorm2d], Type[nn.BatchNorm3d]]:
    """
    A factory function for creating Batch Normalization layers.

    Parameters
    ----------
    dim : int
        Dimension of the image.

    Returns
    -------
    Union[Type[nn.BatchNorm2d], Type[nn.BatchNorm3d]]
        The normalization layer.
    """
    layers = (nn.BatchNorm2d, nn.BatchNorm3d)
    return layers[dim - 2]


def _group_norm_factory(dim: int) -> Type[nn.GroupNorm]:
    """
    A dummy function that returns a Group Normalization layer.

    To match other factory functions.

    Parameters
    ----------
    dim : int
        Dimension of the image.

    Returns
    -------
    Type[nn.GroupNorm]
        The normalization layer.
    """
    return nn.GroupNorm
