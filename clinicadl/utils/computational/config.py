from logging import getLogger

import torch
from pydantic import model_validator
from torch.amp.grad_scaler import GradScaler
from typing_extensions import Self

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import ClinicaDLArgumentError

logger = getLogger("clinicadl.computational_config")


class ComputationalConfig(ClinicaDLConfig):
    """Config class to handle computational parameters."""

    amp: bool = False
    fully_sharded_data_parallel: bool = False
    gpu: bool = False
    non_blocking: bool = True

    @model_validator(mode="after")
    def check_gpu(self) -> Self:
        if self.gpu:
            import torch

            if not torch.cuda.is_available():
                raise ClinicaDLArgumentError(
                    "No GPU is available. To run on CPU, please set gpu to false or add the --no-gpu flag if you use the commandline."
                )
        return self

    @property
    def device(self):
        return torch.device("cuda") if self.gpu else torch.device("cpu")

    def get_scaler(self):
        """TO COMPLETE"""
        return GradScaler(device=self.device.type, enabled=self.amp)
