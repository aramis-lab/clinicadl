import numbers
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Optional

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
    include : Optional[Sequence[str]], default=None
        The key(s) of the ``DataPoints`` to which the transform will be applied. The value associated
        to the key must be a :py:class:`torchio.Image`, a :py:class:`torch.torch.Tensor`,
        a :py:class:`numpy.ndarray`, or a numeric value.

        By default (if ``include=None``), the transform will be applied to all the images,
        i.e. the :py:class:`torchio.Image`, that are not in ``exclude``.
    exclude : Optional[Sequence[str]], default=None
        The key(s) of the ``DataPoints`` to which the transform will **not** be applied.
        ``exclude`` cannot be passed with ``include``.

    Raises
    ------
    ValueError
        If both ``include`` and ``exclude`` are passed.
    """

    def __init__(
        self,
        transform: MonaiTransform,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> None:
        self.transform = transform
        if include and exclude:
            raise ValueError("You cannot pass both 'include' and 'exclude'.")
        self.include = include
        self.exclude = exclude if exclude else []

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={repr(self.transform)}, include={self.include})"

    def __call__(self, datapoint: DataPoint) -> DataPoint:
        """
        Applies the transform to the fields in 'include'.
        """
        datapoint = deepcopy(datapoint)

        for key, value in datapoint.items():
            if key in self.exclude:
                continue
            elif not self.include and not isinstance(value, tio.Image):
                continue
            elif self.include and key not in self.include:
                continue

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

        datapoint.update_attributes()  # so that datapoint.label matches datapoint["label"]

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
        transformed = self._transform(x)
        try:
            return transformed.item()
        except RuntimeError:  # it is no longer a scalar (e.g. one-hot)
            return transformed

    def _transform(self, x: Any) -> torch.Tensor:
        out = self.transform(x)
        if isinstance(out, MetaTensor):
            out = out.as_tensor()
        return out
