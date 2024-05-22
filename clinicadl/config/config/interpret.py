from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Union

from pydantic import BaseModel, field_validator

from clinicadl.interpret.gradients import GradCam, Gradients, VanillaBackProp
from clinicadl.utils.caps_dataset.data import (
    load_data_test,
)
from clinicadl.utils.enum import InterpretationMethod
from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore
from clinicadl.utils.maps_manager.maps_manager import MapsManager  # type: ignore

logger = getLogger("clinicadl.predict_config")


class InterpretConfig(BaseModel):
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
