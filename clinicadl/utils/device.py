import re
from typing import Optional, Union

import torch

DeviceType = Union[str, int, torch.device]


def check_device(device: Optional[DeviceType]) -> Optional[torch.device]:
    """
    Checks if 'device' is in a format accepted by PyTorch for cpu/cuda backend.

    Parameters
    ----------
    device : Optional[DeviceType]
        The device passed by the user.

    Returns
    -------
    Optional[torch.device]
        The device checked and casted in a :py:class:`torch.device` (if not ``None``).
    """
    if isinstance(device, str) and not (
        re.match(r"^cuda:.*", device) or device == "cuda" or device == "cpu"
    ):
        raise ValueError(
            f"If 'device' is a str, it must be 'cpu', 'cuda' or 'cuda:<device-id>'. Got {device}"
        )
    elif device is None:
        return None

    return torch.device(device)
