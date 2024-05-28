from logging import getLogger

from pydantic import BaseModel

from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore

logger = getLogger("clinicadl.predict_config")


class PredictConfig(BaseModel):
    save_tensor: bool = False
    save_latent_tensor: bool = False
    use_labels: bool = True

    def check_output_saving_tensor(self, network_task: str) -> None:
        # Check if task is reconstruction for "save_tensor" and "save_nifti"
        if self.save_tensor and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save tensors if the network task is not reconstruction. Please remove --save_tensor option."
            )
