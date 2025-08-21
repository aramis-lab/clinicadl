from typing import Optional, Sequence, Union

import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import Transform
from monai.utils.type_conversion import (
    convert_data_type,
    convert_to_dst_type,
    convert_to_tensor,
)


class Format(Transform):
    """
    Transform to reformat a :py:class:`torch.Tensor` or a :py:class:`np.ndarray`,
    i.e. to modify its shape and/or its dtype.

    This transform is written to behave like other postprocessing transforms in
    :py:mod:`monai.transforms.post`.

    Parameters
    ----------
    dtype : Optional[torch.dtype], default=None
        The wanted dtype, passed as a :py:class:`torch.dtype`. If ``None``, input's dtype will be kept.
    squeeze : Union[bool, int, Sequence[int]], default=False
        Whether to squeeze the tensor/array, i.e. removing dimension(s) of size 1.
        If ``True``, all such dimensions will be removed. Specific dimension(s) to remove
        can be specified via an ``int`` or a sequence of ``ints``.
    unsqueeze : Optional[int], default=None
        The position where to insert the new dimension. If ``None``, no unsqueezing will be performed.

        .. note::
            Squeezing is performed before unsqueezing.
    """

    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        squeeze: Union[bool, int, Sequence[int]] = False,
        unsqueeze: Optional[int] = None,
    ):
        self.dtype = dtype
        self.squeeze = squeeze
        self.unsqueeze = unsqueeze

    def __call__(
        self,
        img: NdarrayOrTensor,
    ) -> NdarrayOrTensor:
        img: torch.Tensor = convert_to_tensor(img)
        img_t, *_ = convert_data_type(img, torch.Tensor)

        if self.squeeze is True:
            img_t.squeeze_()
        elif self.squeeze is not False:
            img_t.squeeze_(self.squeeze)

        if self.unsqueeze is not None:
            img_t.unsqueeze_(self.unsqueeze)

        out, *_ = convert_to_dst_type(img_t, img, dtype=self.dtype)

        return out
