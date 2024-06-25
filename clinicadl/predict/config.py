from logging import getLogger

from pydantic import BaseModel, ConfigDict

from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.maps_manager.config import (
    MapsManagerConfig as MapsManagerBaseConfig,
)
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore
from clinicadl.validation.cross_validation import CrossValidationConfig
from clinicadl.validation.validation import ValidationConfig

logger = getLogger("clinicadl.predict_config")


class PredictConfig(BaseModel):
    """Config class to perform Transfer Learning."""

    maps_manager: MapsManagerConfig
    data: DataConfig
    validation: ValidationConfig
    cross_validation: CrossValidationConfig
    computational: ComputationalConfig
    dataloader: DataLoaderConfig

    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, **kwargs):
        super().__init__(
            maps_manager=kwargs,
            computational=kwargs,
            cross_validation=kwargs,
            data=kwargs,
            dataloader=kwargs,
            validation=kwargs,
        )
