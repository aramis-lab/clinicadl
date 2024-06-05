from logging import getLogger

from pydantic import BaseModel, ConfigDict
from pydantic.types import PositiveInt

from clinicadl.utils.enum import Sampler

logger = getLogger("clinicadl.dataloader_config")


class DataLoaderConfig(BaseModel):  # TODO : put in data/splitter module
    """Config class to configure the DataLoader."""

    batch_size: PositiveInt = 8
    n_proc: PositiveInt = 2
    sampler: Sampler = Sampler.RANDOM
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)
