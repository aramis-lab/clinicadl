from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import torch
import torchio as tio
from monai.utils.type_conversion import (
    convert_data_type,
    convert_to_dst_type,
    convert_to_tensor,
)

from clinicadl.utils.dtype import DtypeLike
from clinicadl.utils.numerics import merge_numerics

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint


class Format(tio.Transform):
    """
    Transform to modify the shape or the type of some values in a :py:class:`~clinicadl.data.structures.DataPoint`.

    To be transformed, the values are expected to be :py:class:`torch.Tensor` or a :py:class:`np.ndarray`.

    This transform inherits from :py:class:`torchio.Transforms`, and therefore any argument accepted
    by this parent class can be passed via a keyword argument here. Particularly, you may be interested in
    ``include`` or ``exclude`` to specify the keys whose values should be modified, and ``copy`` to specify
    if the output ``DataPoint`` should be the same object as the input or a deepcopy.

    Parameters
    ----------
    dtype : Optional[torch.dtype], default=None
        The wanted dtype, passed as a :py:class:`torch.dtype`, a :py:class:`numpy.dtype`, or a ``str`` (e.g., "float32", "int64").
        If ``None``, input's dtype will be kept.
    squeeze : Union[bool, int, Sequence[int]], default=False
        Whether to squeeze the tensor/array, i.e. removing dimension(s) of size 1.
        If ``True``, all such dimensions will be removed. Specific dimension(s) to remove
        can be specified via an ``int`` or a sequence of ``ints``.
    unsqueeze : Optional[int], default=None
        The position where to insert the new dimension. If ``None``, no unsqueezing will be performed.

        .. note::
            Squeezing is performed before unsqueezing.

    **kwargs: Any
        Any keyword argument accepted by :py:class:`torchio.Transform`.

    Example
    -------
    .. code-block::

        from clinicadl.transforms import Format
        from clinicadl.data.structures.examples import ColinDataPoint
        import numpy as np

        data = ColinDataPoint(age=55.0, array=np.array([1, 2]))

    .. code-block::

        >>> Format(dtype="int64", include=["age"])(data).age
        55
        >>> Format(unsqueeze=1, include=["array"])(data).array
        [[1]
         [2]]
    """

    def __init__(
        self,
        dtype: Optional[DtypeLike] = None,
        squeeze: Union[bool, int, Sequence[int]] = False,
        unsqueeze: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dtype = dtype
        self.squeeze = squeeze
        self.unsqueeze = unsqueeze
        self.args_names = ["dtype", "squeeze", "unsqueeze"]

    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        for key in datapoint.get_keys(include=self.include, exclude=self.exclude):
            datapoint[key] = self._format(datapoint[key])

        return datapoint

    def _format(
        self,
        value: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            value = np.array(value)
        value_t, *_ = convert_data_type(value, output_type=torch.Tensor)

        if self.squeeze is True:
            value_t.squeeze_()
        elif self.squeeze is not False:
            value_t.squeeze_(self.squeeze)

        if self.unsqueeze is not None:
            value_t.unsqueeze_(self.unsqueeze)

        out, *_ = convert_to_dst_type(value_t, value, dtype=self.dtype)

        return out


class MergeFields(tio.Transform):
    """
    Transform to merge several values of a :py:class:`~clinicadl.data.structures.DataPoint`.

    The result of the merger depends on the type of values:

    - :py:class:`torch.Tensor` and :py:class:`np.ndarray` are stacked along a new dimension; they are
      expected to all have the same shape;
    - :py:class:`torchio.Image` are concatenated along the channel dimension; they are
      expected to all have the same spatial shape;
    - ``lists`` and ``tuples`` are concatenated;
    - otherwise, the values are just put in a list.

    This transform inherits from :py:class:`torchio.Transforms`, and therefore any argument accepted
    by this parent class can be passed via a keyword argument here. Particularly, you may be interested in
    in ``copy`` to specify if the output ``DataPoint`` should be the same object as the input or a deepcopy.

    Parameters
    ----------
    *keys : str
        The keys whose values should be merged.
    output_key : str
        The name of key in the :py:class:`~clinicadl.data.structures.DataPoint` corresponding
        to the output of the merger.
    **kwargs: Any
        Any keyword argument accepted by :py:class:`torchio.Transform`.

    Example
    -------
    .. code-block::

        from clinicadl.transforms import MergeFields
        from clinicadl.data.structures.examples import ColinDataPoint
        import numpy as np

        data = ColinDataPoint(age=55, sex="M", array_1=np.zeros(2), array_2=np.ones(2))

    .. code-block::

        >>> MergeFields("age", "sex", output_key="label")(data).label
        [55, 'M']
        >>> MergeFields("array_1", "array_2", output_key="label")(data)
        [[0. 0.]
         [1. 1.]]
    """

    def __init__(self, *keys: str, output_key: str, **kwargs):
        super().__init__(**kwargs)
        self.keys = keys
        self.output_key = output_key
        self.args_names = ["keys", "output_key"]

    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        values = [datapoint[key] for key in self.keys]

        datapoint[self.output_key] = merge_numerics(values)

        return datapoint
