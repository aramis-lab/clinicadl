from logging import getLogger
from pathlib import Path
from typing import Dict, Enum, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, PrivateAttr, field_validator

from clinicadl.interpret.gradients import GradCam, VanillaBackProp
from clinicadl.utils.caps_dataset.data import (
    get_transforms,
    load_data_test,
    return_dataset,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore
from clinicadl.utils.maps_manager.maps_manager import MapsManager  # type: ignore

logger = getLogger("clinicadl.predict_config")


class InterpretationMethod(str, Enum):
    """Possible interpretation method in clinicaDL."""

    GRADIENTS = "gradients"
    GRAD_CAM = "grad-cam"


class PredictInterpretConfig(BaseModel):
    maps_dir: Path
    data_group: str
    caps_directory: Path
    tsv_path: Path
    selection_metrics: list[str] = ["loss"]
    split_list: list[int] = []
    diagnoses: list[str] = []
    multi_cohort: bool = False
    batch_size: int = 8
    n_proc: int = 0
    gpu: bool = True
    amp: bool = False
    overwrite: bool = False
    save_nifti: bool = False
    skip_leak_check: bool = False

    @field_validator("selection_metrics", "split_list", "diagnoses", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

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
    method: InterpretationMethod = InterpretationMethod.GRADIENTS
    target_node: int = 0
    save_individual: bool = False
    overwrite_name: bool = False
    level: int = 1
    _method_dict: dict = PrivateAttr(
        default={"gradients": VanillaBackProp, "grad-cam": GradCam}
    )

    @field_validator("method", "_method_dict", check_fields=False)
    def validator_method(cls, _method, _method_dict):
        if _method not in _method_dict:
            raise NotImplementedError(
                f"Interpretation method {_method} is not implemented. "
                f"Please choose in {_method_dict.keys()}"
            )

    def create_groupe_df(self):
        group_df = None
        if self.tsv_path is not None:
            group_df = load_data_test(
                self.tsv_path,
                self.diagnoses,
                multi_cohort=self.multi_cohort,
            )
        return group_df

    def get_method(self):
        return self._method_dict[self.method]


class PredictConfig(PredictInterpretConfig):
    label: str = ""
    save_tensor: bool = False
    save_latent_tensor: bool = False
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

    def check_label(self, _label: str):
        if not self.label:
            self.label = _label
