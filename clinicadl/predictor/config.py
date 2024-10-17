from logging import getLogger
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, computed_field

from clinicadl.caps_dataset.data_config import DataConfig as DataBaseConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.maps_manager.config import (
    MapsManagerConfig as MapsManagerBaseConfig,
)
from clinicadl.maps_manager.maps_manager import MapsManager
from clinicadl.predictor.validation import ValidationConfig
from clinicadl.splitter.config import SplitConfig
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.enum import Task
from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore

logger = getLogger("clinicadl.predict_config")


class MapsManagerConfig(MapsManagerBaseConfig):
    save_tensor: bool = False
    save_latent_tensor: bool = False

    def check_output_saving_tensor(self, network_task: str) -> None:
        # Check if task is reconstruction for "save_tensor" and "save_nifti"
        if self.save_tensor and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save tensors if the network task is not reconstruction. Please remove --save_tensor option."
            )


class DataConfig(DataBaseConfig):
    use_labels: bool = True


class PredictConfig(BaseModel):
    """Config class to perform Transfer Learning."""

    maps_manager: MapsManagerConfig
    data: DataConfig
    validation: ValidationConfig
    computational: ComputationalConfig
    dataloader: DataLoaderConfig
    split: SplitConfig
    transforms: TransformsConfig

    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, **kwargs):
        super().__init__(
            maps_manager=kwargs,
            computational=kwargs,
            dataloader=kwargs,
            data=kwargs,
            split=kwargs,
            validation=kwargs,
            transforms=kwargs,
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

        if self.split.split is None and self.split.n_splits == 0:
            from clinicadl.splitter.split_utils import find_splits

            self.split.split = find_splits(self.maps_manager.maps_dir)
