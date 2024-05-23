from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator

from clinicadl.interpret.gradients import GradCam, Gradients, VanillaBackProp
from clinicadl.utils.caps_dataset.data import (
    load_data_test,
)
from clinicadl.utils.enum import InterpretationMethod
from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore
from clinicadl.utils.maps_manager.maps_manager import MapsManager  # type: ignore

logger = getLogger("clinicadl.predict_config")


class MapsManagerConfig(BaseModel):
    maps_dir: Path
    data_group: str
    overwrite: bool = False
    save_nifti: bool = False

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    def check_output_saving_nifti(self, network_task: str) -> None:
        # Check if task is reconstruction for "save_tensor" and "save_nifti"
        if self.save_nifti and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save nifti if the network task is not reconstruction. Please remove --save_nifti option."
            )
