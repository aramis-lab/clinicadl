from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torchio as tio
from monai.data import MetaTensor
from monai.transforms import Transform as MonaiTransform

if TYPE_CHECKING:
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
    copy : bool, default=False
        Whether to make a deepcopy of the input before applying the transforms.

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
        copy: bool = False,
    ) -> None:
        self.transform = transform
        if include and exclude:
            raise ValueError("You cannot pass both 'include' and 'exclude'.")
        self.include = include
        self.exclude = exclude if exclude else []
        self.copy = copy

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={repr(self.transform)}, include={self.include})"

    def __call__(self, datapoint: DataPoint) -> DataPoint:
        """
        Applies the transform to the fields in 'include'.
        """
        if self.copy:
            datapoint = deepcopy(datapoint)

        for key, value in datapoint.items():
            if key in self.exclude:
                continue
            elif not self.include and not isinstance(value, tio.Image):
                continue
            elif self.include and key not in self.include:
                continue

            value = datapoint[key]

            if isinstance(value, tio.Image):
                self._transform_tio_image(value)
            else:
                if isinstance(value, torch.Tensor):
                    transform = self._transform
                else:
                    transform = self._transform_array_like

                try:
                    datapoint[key] = transform(value)
                except Exception as e:
                    raise Exception(
                        f"An error occurred while transforming the field '{key}'."
                    ) from e

        return datapoint

    def _transform_tio_image(self, x: tio.Image) -> None:
        x.set_data(self._transform(x.tensor))

    def _transform_array_like(self, x: np.typing.ArrayLike) -> np.ndarray:
        return self._transform(np.array(x)).numpy()

    def _transform(self, x: Any) -> torch.Tensor:
        out = self.transform(x)
        if isinstance(out, MetaTensor):
            out = out.as_tensor()
        return out
