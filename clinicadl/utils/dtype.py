from typing import Optional, Union

import numpy as np
import torch

DtypeLike = Union[np.dtype, torch.dtype, str]


def read_dtype(dtype_str: Optional[str]) -> Optional[DtypeLike]:
    """
    To read a serialized dtype.

    The function tries to convert the dtype to :py:class:`numpy.dtype` or a :py:class:`torch.dtype`.

    Parameters
    ----------
    dtype_str : Optional[str]
        The dtype as a string.

    Returns
    -------
    Union[np.dtype, torch.dtype, str]
        The dtype as a :py:class:`numpy.dtype`, :py:class:`torch.dtype`, or
        a string.
    """
    if dtype_str is None:
        return None

    if dtype_str.startswith("torch."):
        return getattr(torch, dtype_str.split(".", 1)[1])
    elif dtype_str.startswith("np.") or dtype_str.startswith("numpy."):
        return getattr(np, dtype_str.split(".", 1)[1])

    return dtype_str
