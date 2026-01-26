from typing import Optional

import torch
from pydantic import NonNegativeInt, PositiveInt, field_validator
from torch.amp.grad_scaler import GradScaler

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import ClinicaDLArgumentError


class ComputationalConfig(ClinicaDLConfig):
    """
    Config class to define computational parameters.

    Parameters
    ----------
    gpu : bool, default=False
        Whether to use a GPU.
    non_blocking : bool, default=True
        Behavior to adopt when sending data or the model to a GPU:
        "When ``non_blocking`` is set to ``True``, [...] attempts to perform the
        conversion asynchronously with respect to the host, if possible.
        This asynchronous behavior applies to both pinned and pageable memory."
        (see :torch:`PyTorch documentation <generated/torch.Tensor.to.html>`)
    amp : bool, default=True
        Whether to use :py:mod:`Automatic Mixed Precision <torch.amp>`.
    channels_last : bool, default=True
        Whether to use `Channels Last Memory Format <https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_
        when possible.
    checkpoint_every : PositiveInt, default=1
        Save checkpoint every x epochs. If your training fails, ``ClinicaDL`` will resume
        from the last checkpoint.
    seed : Optional[NonNegativeInt], default=None
        To seed the randomness in your training.
    """

    gpu: bool = True
    non_blocking: bool = True
    amp: bool = True
    channels_last: bool = True
    checkpoint_every: PositiveInt = 1
    seed: Optional[NonNegativeInt] = None

    @field_validator("gpu", mode="after")
    @classmethod
    def _check_gpu(cls, value: bool) -> bool:
        """
        Check if GPU is indeed available.
        """
        if value:
            import torch

            if not torch.cuda.is_available():
                raise ClinicaDLArgumentError("No GPU available!.")
        return value

    @property
    def device(self):
        """
        The device, represented as a :py:class:`torch.device`.
        """
        return torch.device("cuda") if self.gpu else torch.device("cpu")

    def get_scaler(self) -> GradScaler:
        """
        To get the :py:class:`torch.amp.GradScaler`.

        Returns
        -------
        GradScaler
            The gradient scaler.
        """
        return GradScaler(device=self.device.type, enabled=self.amp)
