import os
from typing import Optional

import torch
from pydantic import NonNegativeInt, field_validator
from torch.amp.grad_scaler import GradScaler

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.seed import DETERMINISTIC, GLOBAL_SEED


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
    seed : Optional[NonNegativeInt], default=None
        Global seed to control the randomness. If ``None``, ``ComputationalConfig`` will look for a global seed set
        with :py:func:`clinicadl.utils.seed.seed_everything`. If passed, it will override any global seed.
    deterministic : Optional[bool], default=None
        Whether to configure PyTorch's operations in deterministic mode. If ``None``, ``ComputationalConfig`` will look for a global configuration set
        with :py:func:`clinicadl.utils.seed.seed_everything`. If passed, it will override any global configuration.
    """

    gpu: bool = True
    non_blocking: bool = True
    amp: bool = True
    channels_last: bool = True
    seed: Optional[NonNegativeInt] = None
    deterministic: Optional[bool] = None

    def check_device(self) -> None:
        """
        Checks that the requested device is available.
        """
        if self.gpu:
            assert torch.cuda.is_available(), "No GPU with CUDA available."

    @field_validator("seed", mode="after")
    @classmethod
    def _check_global_seed(
        cls, seed: Optional[NonNegativeInt]
    ) -> Optional[NonNegativeInt]:
        """If no seed, look for a global seed."""
        if seed is None and (glob_seed := os.environ.get(GLOBAL_SEED)) is not None:
            return int(glob_seed)
        return seed

    @field_validator("deterministic", mode="after")
    @classmethod
    def _check_deterministic(cls, deterministic: Optional[bool]) -> bool:
        """If no deterministic arg, look for a global configuration."""
        if deterministic is None:
            return bool(os.environ.get(DETERMINISTIC))
        return deterministic

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
