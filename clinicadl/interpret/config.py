from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from clinicadl.caps_dataset.data_config import DataConfig as DataBaseConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.interpret.gradients import GradCam, Gradients, VanillaBackProp
from clinicadl.maps_manager.config import MapsManagerConfig
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.enum import InterpretationMethod
from clinicadl.validation.cross_validation import CrossValidationConfig
from clinicadl.validation.validation import ValidationConfig

logger = getLogger("clinicadl.interpret_config")


class DataConfig(DataBaseConfig):
    caps_directory: Optional[Path] = None


class InterpretBaseConfig(BaseModel):
    name: str
    method: InterpretationMethod = InterpretationMethod.GRADIENTS
    target_node: int = 0
    save_individual: bool = False
    overwrite_name: bool = False
    level: Optional[int] = 1

    @field_validator("level", mode="before")
    def chek_level(cls, v):
        if v < 1:
            raise ValueError(
                f"You must set the level to a number bigger than 1. ({v} < 1)"
            )

    def get_method(self) -> Gradients:
        if self.method == InterpretationMethod.GRADIENTS:
            return VanillaBackProp
        elif self.method == InterpretationMethod.GRAD_CAM:
            return GradCam
        else:
            raise ValueError(f"The method {self.method.value} is not implemented")


class InterpretConfig(BaseModel):
    """Config class to perform Transfer Learning."""

    maps_manager: MapsManagerConfig
    interpret: InterpretBaseConfig
    data: DataConfig
    validation: ValidationConfig
    cross_validation: CrossValidationConfig
    computational: ComputationalConfig
    dataloader: DataLoaderConfig

    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, **kwargs):
        super().__init__(
            maps_manager=kwargs,
            interpret=kwargs,
            validation=kwargs,
            cross_validation=kwargs,
            data=kwargs,
            dataloader=kwargs,
            computational=kwargs,
        )
