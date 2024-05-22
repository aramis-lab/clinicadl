from logging import getLogger

from pydantic import BaseModel, ConfigDict

logger = getLogger("clinicadl.computational_config")


class ComputationalConfig(BaseModel):
    """Config class to handle computational parameters."""

    amp: bool = False
    fully_sharded_data_parallel: bool = False
    gpu: bool = True
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
