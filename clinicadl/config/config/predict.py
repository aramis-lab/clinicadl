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


class PredictConfig(BaseModel):
    label: str = ""
    save_tensor: bool = False
    save_latent_tensor: bool = False
    use_labels: bool = True

    def is_given_label_code(self, _label: str, _label_code: Union[str, Dict[str, int]]):
        return (
            self.label is not None
            and self.label != ""
            and self.label != _label
            and _label_code == "default"
        )

    def check_label(self, _label: str):
        if not self.label:
            self.label = _label

    def check_output_saving_tensor(self, network_task: str) -> None:
        # Check if task is reconstruction for "save_tensor" and "save_nifti"
        if self.save_tensor and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save tensors if the network task is not reconstruction. Please remove --save_tensor option."
            )
