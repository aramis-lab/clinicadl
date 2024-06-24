from logging import getLogger

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from clinicadl.utils.exceptions import ClinicaDLArgumentError

logger = getLogger("clinicadl.computational_config")


class ComputationalConfig(BaseModel):
    """Config class to handle computational parameters."""

    amp: bool = False
    fully_sharded_data_parallel: bool = False
    gpu: bool = True
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def check_gpu(self) -> Self:
        if self.gpu:
            import torch

            if not torch.cuda.is_available():
                raise ClinicaDLArgumentError(
                    "No GPU is available. To run on CPU, please set gpu to false or add the --no-gpu flag if you use the commandline."
                )
        elif self.amp:
            raise ClinicaDLArgumentError(
                "AMP is designed to work with modern GPUs. Please add the --gpu flag."
            )
        return self
