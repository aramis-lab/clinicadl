import itertools
from typing import Any, Sequence, TypeVar, overload

import numpy as np
import torch
import torchio as tio

from .variables import SPACING_RTOL

T = TypeVar("T")
NumericT = TypeVar("NumericT", tio.Image, torch.Tensor, np.ndarray)


@overload
def merge_numerics(values: Sequence[NumericT]) -> NumericT:
    ...


@overload
def merge_numerics(values: Sequence[T]) -> list[T]:
    ...


def merge_numerics(values: Sequence[Any]) -> Any:
    """
    Tries to merge elements of a sequence depending on their types:

    - :py:class:`torch.Tensor` and :py:class:`np.ndarray` are stacked along a new dimension;
    - :py:class:`torchio.Image` are concatenated along the channel dimension;
    - ``lists`` and ``tuples`` are concatenated;
    - otherwise, the sequence is returned as a list.

    Parameters
    ----------
    values : Sequence[Any]
        The values to merge.

    Returns
    -------
    Any
        The result of the merger, or the input value if merging was not possible.
    """
    error_message = "An error occurred when merging the values, probably because the have different shapes."

    if all(isinstance(value, np.ndarray) for value in values):
        try:
            return np.stack(values)
        except ValueError as e:
            raise RuntimeError(error_message) from e

    elif all(isinstance(value, (torch.Tensor, np.ndarray)) for value in values):
        values = [
            torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for value in values
        ]

    if all(isinstance(value, torch.Tensor) for value in values):
        try:
            return torch.stack(values)
        except RuntimeError as e:
            raise RuntimeError(error_message) from e

    elif all(isinstance(value, tio.Image) for value in values):
        return concat_images(values)

    elif all(isinstance(value, (tuple, list)) for value in values):
        return list(itertools.chain(*values))

    else:
        return list(values)


def concat_images(images: Sequence[tio.Image]) -> tio.Image:
    """
    Concatenates :py:class:`TorchIO's images <torchio.Image>` into
    a single image.

    Parameters
    ----------
    images : Sequence[tio.Image]
        The images to concatenate.

    Returns
    -------
    tio.Image
        The output image.
    """
    _check_spacing(images)
    _check_shape(images)

    tensor = torch.cat([image.tensor for image in images], dim=0)
    affine = images[0].affine
    if all(isinstance(value, tio.LabelMap) for value in images):
        image = tio.LabelMap(tensor=tensor, affine=affine)
    else:
        image = tio.ScalarImage(tensor=tensor, affine=affine)

    return image


def _check_spacing(images: Sequence[tio.Image]) -> None:
    """
    Check if spacing is consistent before concatenating.
    """
    ref_spacing = images[0].spacing
    for image in images[1:]:
        if not np.isclose(ref_spacing, image.spacing, rtol=SPACING_RTOL).all():
            raise RuntimeError(
                "Trying to concatenate images with different voxel spacings!"
            )


def _check_shape(images: Sequence[tio.Image]) -> None:
    """
    Check if spatial shape is consistent before concatenating.
    """
    ref_shape = images[0].spatial_shape
    for image in images[1:]:
        if image.spatial_shape != ref_shape:
            raise RuntimeError(
                "Trying to concatenate images with different spatial shapes!"
            )
