from typing import Type, Union

import torch.nn as nn

from clinicadl.utils.enum import BaseEnum


class Normalization(str, BaseEnum):
    """Available normalization layers in ClinicaDL."""

    BATCH = "BatchNorm"
    GROUP = "GroupNorm"
    INSTANCE = "InstanceNorm"


class NormLayer:
    """Factory object for creating Normalization layers."""

    def __new__(
        cls, normalization: Union[str, Normalization], dim: int
    ) -> Type[nn.Module]:
        """
        Creates a Normalization layer.

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
    layers = (nn.InstanceNorm2d, nn.InstanceNorm3d)
    return layers[dim - 2]


def _batch_norm_factory(dim: int) -> Union[Type[nn.BatchNorm2d], Type[nn.BatchNorm3d]]:
    layers = (nn.BatchNorm2d, nn.BatchNorm3d)
    return layers[dim - 2]


def _group_norm_factory(dim: int) -> Type[nn.GroupNorm]:
    return nn.GroupNorm
