from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path

import nibabel as nib
import torch

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import ExtractionMethod

logger = getLogger("clinicadl.preprocessing_config")

NII_GZ = ".nii.gz"
PT = ".pt"


class BaseExtraction(ClinicaDLConfig):
    """
    Abstract config class for the Extraction procedure.
    """

    extract_method: ExtractionMethod
    use_uncropped_image: bool = True
    # save_features: bool = False
    # extract_json: str = f"extract_{int(time())}.json"

    # @field_validator("extract_json", mode="before")

    def extract_image(self, input_img: Path) -> torch.Tensor:
        """Loads a NIfTI image and returns it as a float32 tensor."""
        image_array = nib.loadsave.load(input_img).get_fdata(dtype="float32")  # type: ignore
        return torch.from_numpy(image_array).unsqueeze(0).float()

    @abstractmethod
    def extract_tensor(
        self,
        image_tensor: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        """Extracts specific data from an image tensor."""
        pass

    @abstractmethod
    def extract_path(self, image_path, index):
        """Defines path for saving extracted elements."""
        pass

    @abstractmethod
    def extract(self, nii_path: Path):
        """Performs extraction based on the implemented method."""
        pass

    @abstractmethod
    def num_elem_per_image(self, image: torch.Tensor) -> int:
        """Returns the number of extracted elements per image."""
        pass
