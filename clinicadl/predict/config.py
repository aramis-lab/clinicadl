from logging import getLogger

from clinicadl.caps_dataset.data_config import DataConfig as DataBaseConfig
from clinicadl.caps_dataset.dataloader_config import DataLoaderConfig
from clinicadl.maps_manager.config import (
    MapsManagerConfig as MapsManagerBaseConfig,
)
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore
from clinicadl.validation.cross_validation import CrossValidationConfig
from clinicadl.validation.validation import ValidationConfig

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


class PredictConfig(
    MapsManagerConfig,
    DataConfig,
    ValidationConfig,
    CrossValidationConfig,
    ComputationalConfig,
    DataLoaderConfig,
):
    """Config class to perform Transfer Learning."""
