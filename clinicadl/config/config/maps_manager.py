from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore
from clinicadl.utils.maps_manager.maps_manager import MapsManager  # type: ignore

logger = getLogger("clinicadl.predict_config")


class MapsManagerConfig(BaseModel):
    maps_dir: Path
    data_group: Optional[str] = None
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

    def adapt_config_with_maps_manager_info(self, maps_manager: MapsManager):
        if not self.split_list:
            self.split_list = maps_manager._find_splits()
        logger.debug(f"List of splits {self.split_list}")

        if self.diagnoses is None or len(self.diagnoses) == 0:
            self.diagnoses = maps_manager.diagnoses

        if not self.batch_size:
            self.batch_size = maps_manager.batch_size

        if not self.n_proc:
            self.n_proc = maps_manager.n_proc
