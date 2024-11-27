from logging import getLogger
from pathlib import Path
from typing import Tuple

import torch

from clinicadl.utils.enum import ExtractionMethod

from .base import BaseExtraction

logger = getLogger("clinicadl.preprocessing_config")

NII_GZ = ".nii.gz"
PT = ".pt"


class Image(BaseExtraction):
    """
    Configuration class for full image extraction as a single tensor.
    """

    extract_method: ExtractionMethod = ExtractionMethod.IMAGE

    def extract(self, nii_path: Path) -> list[Tuple[Path, torch.Tensor]]:
        """Extracts the full image as a single tensor file and saves it."""
        image_tensor = self.extract_image(nii_path)
        output_file = Path(Path(nii_path.stem).stem + PT), image_tensor.clone()
        return [output_file]

    def extract_tensor(
        self,
        image_tensor: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        return image_tensor

    def extract_path(self, image_path, index):
        return image_path

    def num_elem_per_image(self, image: torch.Tensor):
        return 1
