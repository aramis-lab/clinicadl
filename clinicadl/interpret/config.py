from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator

from clinicadl.caps_dataset.data_config import DataConfig as DataBaseConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.interpret.gradients import GradCam, Gradients, VanillaBackProp
from clinicadl.maps_manager.config import MapsManagerConfig as MapsManagerConfigBase
from clinicadl.maps_manager.maps_manager import MapsManager
from clinicadl.predictor.validation import ValidationConfig
from clinicadl.splitter.config import SplitConfig
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.enum import InterpretationMethod
from clinicadl.utils.exceptions import ClinicaDLArgumentError

logger = getLogger("clinicadl.interpret_config")


class MapsManagerConfig(MapsManagerConfigBase):
    save_tensor: bool = False

    def check_output_saving_tensor(self, network_task: str) -> None:
        # Check if task is reconstruction for "save_tensor" and "save_nifti"
        if self.save_tensor and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save tensors if the network task is not reconstruction. Please remove --save_tensor option."
            )


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
    data: DataConfig
    validation: ValidationConfig
    computational: ComputationalConfig
    dataloader: DataLoaderConfig
    split: SplitConfig
    interpret: InterpretBaseConfig

    def __init__(self, **kwargs):
        super().__init__(
            maps_manager=kwargs,
            computational=kwargs,
            dataloader=kwargs,
            data=kwargs,
            split=kwargs,
            validation=kwargs,
            transforms=kwargs,
            interpret=kwargs,
        )

    def _update(self, config_dict: Dict[str, Any]) -> None:
        """Updates the configs with a dict given by the user."""
        self.data.__dict__.update(config_dict)
        self.split.__dict__.update(config_dict)
        self.validation.__dict__.update(config_dict)
        self.maps_manager.__dict__.update(config_dict)
        self.split.__dict__.update(config_dict)
        self.computational.__dict__.update(config_dict)
        self.dataloader.__dict__.update(config_dict)
        self.transforms.__dict__.update(config_dict)
        self.interpret.__dict__.update(config_dict)

    def adapt_with_maps_manager_info(self, maps_manager: MapsManager):
        self.maps_manager.check_output_saving_nifti(maps_manager.network_task)
        self.data.diagnoses = (
            maps_manager.diagnoses
            if self.data.diagnoses is None or len(self.data.diagnoses) == 0
            else self.data.diagnoses
        )

        self.dataloader.batch_size = (
            maps_manager.batch_size
            if not self.dataloader.batch_size
            else self.dataloader.batch_size
        )
        self.dataloader.n_proc = (
            maps_manager.n_proc
            if not self.dataloader.n_proc
            else self.dataloader.n_proc
        )

        self.split.adapt_cross_val_with_maps_manager_info(maps_manager)
        self.maps_manager.check_output_saving_tensor(maps_manager.network_task)

        self.transforms = TransformsConfig(
            normalize=maps_manager.normalize,
            data_augmentation=maps_manager.data_augmentation,
            size_reduction=maps_manager.size_reduction,
            size_reduction_factor=maps_manager.size_reduction_factor,
        )
