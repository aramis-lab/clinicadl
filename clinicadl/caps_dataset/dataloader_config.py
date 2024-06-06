from logging import getLogger

from pydantic import BaseModel, ConfigDict
from pydantic.types import PositiveInt

from clinicadl.utils.enum import Sampler
from clinicadl.utils.maps_manager.maps_manager import MapsManager

logger = getLogger("clinicadl.dataloader_config")


class DataLoaderConfig(BaseModel):  # TODO : put in data/splitter module
    """Config class to configure the DataLoader."""

    batch_size: PositiveInt = 8
    n_proc: PositiveInt = 2
    sampler: Sampler = Sampler.RANDOM
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    def adapt_dataloader_with_maps_manager_info(self, maps_manager: MapsManager):
        # TEMPORARY
        if not self.batch_size:
            self.batch_size = maps_manager.batch_size

        if not self.n_proc:
            self.n_proc = maps_manager.n_proc
