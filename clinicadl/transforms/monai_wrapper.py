import numbers
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import torch
import torchio as tio
from monai.data import MetaTensor
from monai.transforms import Transform as MonaiTransform
from numpy import ndarray

from clinicadl.data.structures import DataPoint


class MonaiTransformWrapper:
    """Converts a transform from ``MONAI`` to
    a transform compatible with ``ClinicaDL``, i.e. a
    transform that works with a :py:class:`clinicadl.data.structures.DataPoint`.

    Parameters
    ----------
    transform : MonaiTransform
        A :py:class:`monai.transforms.Transform`.
    include : Sequence[str]
        The key(s) of the ``DataPoints`` on which to apply the transform. The value associated
        to the key must be a :py:class:`torchio.Image`, a :py:class:`torch.torch.Tensor`,
        a :py:class:`numpy.ndarray`, or a numeric value.
    """

    def __init__(self, transform: MonaiTransform, include: Sequence[str]) -> None:
        self.transform = transform
        self.include = include

    def __call__(self, datapoint: DataPoint) -> DataPoint:
        """
        Applies the transform to the fields in 'include'.
        """
        datapoint = deepcopy(datapoint)

        for key in self.include:
            value = datapoint[key]
            self._check_type(key, value)

            if isinstance(value, tio.Image):
                self._transform_tio_image(value)
            else:
                if isinstance(value, torch.Tensor):
                    transform = self._transform
                elif isinstance(value, ndarray):
                    transform = self._transform_ndarray
                elif isinstance(value, numbers.Number):
                    transform = self._transform_numeric

                datapoint[key] = transform(value)

        return datapoint

    def _check_type(self, key: str, value: Any) -> None:
        """
        Checks that we have a type accepted by MONAI.
        """
        if not isinstance(value, (tio.Image, torch.Tensor, ndarray, numbers.Number)):
            raise TypeError(
                f"To apply '{self.transform.__class__.__name__}', '{key}' must be a torchio.Image, a torch.Tensor, a numpy.ndarray, "
                f"or a numeric value. Got a {type(value)}"
            )

    def _transform_tio_image(self, x: tio.Image) -> None:
        x.set_data(self._transform(x.tensor))

    def _transform_ndarray(self, x: ndarray) -> ndarray:
        return self._transform(x).numpy()

    def _transform_numeric(self, x: numbers.Number) -> numbers.Number:
        return self._transform(x).item()

    def _transform(self, x: Any) -> torch.Tensor:
        out = self.transform(x)
        if isinstance(out, MetaTensor):
            out = out.as_tensor()
        return out
