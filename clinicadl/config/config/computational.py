from logging import getLogger

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from clinicadl.utils.cmdline_utils import check_gpu
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
    def validator_gpu(self) -> Self:
        if self.gpu:
            check_gpu()
        elif self.amp:
            raise ClinicaDLArgumentError(
                "AMP is designed to work with modern GPUs. Please add the --gpu flag."
            )
        return self
