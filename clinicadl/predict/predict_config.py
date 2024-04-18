from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore
from clinicadl.utils.maps_manager.maps_manager import MapsManager  # type: ignore

logger = getLogger("clinicadl.predict_config")


class PredictInterpretConfig(BaseModel):
    maps_dir: Path
    data_group: str
    caps_directory: Optional[Path]
    tsv_path: Optional[Path]
    selection_metrics: List[str] = ["loss"]
    diagnoses: Optional[List[str]]
    multi_cohort: bool = False
    batch_size: int = 8
    n_proc: int = 0
    gpu: bool = True
    amp: bool = False
    overwrite: bool = False
    save_nifti: bool = False

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


class InterpretConfig(PredictInterpretConfig):
    name: str
    method: str = "gradients"
    target_node: int = 0
    save_individual: bool = False
    overwrite_name: bool = False
    level: int = 1


class PredictConfig(PredictInterpretConfig):
    label: Optional[str] = None
    save_tensor: bool = False
    save_latent_tensor: bool = False
    skip_leak_check: bool = False
    split_list: Optional[List[int]]
    use_labels: bool = True

    def check_output_saving(self, network_task: str) -> None:
        # Check if task is reconstruction for "save_tensor" and "save_nifti"
        if self.save_tensor and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save tensors if the network task is not reconstruction. Please remove --save_tensor option."
            )
        if self.save_nifti and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save nifti if the network task is not reconstruction. Please remove --save_nifti option."
            )

    def is_given_label_code(self, _label: str, _label_code: Union[str, Dict[str, int]]):
        return not (
            self.label is not None and self.label != _label and _label_code == "default"
        )
