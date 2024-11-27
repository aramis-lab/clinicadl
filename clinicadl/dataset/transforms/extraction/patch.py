from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import torch

from clinicadl.utils.enum import ExtractionMethod

from .base import BaseExtraction

logger = getLogger("clinicadl.preprocessing_config")

NII_GZ = ".nii.gz"
PT = ".pt"


class Patch(BaseExtraction):
    """
    Configuration class for patch extraction from an image with defined patch size and stride.
    """

    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH

    def num_elem_per_image(self, image: torch.Tensor):
        """Returns the total number of patches generated from the image."""
        return self.create_patches(image).shape[0]

    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """Extracts patches from a NIfTI image tensor."""

        image_tensor = self.extract_image(nii_path)
        patches_tensor = self.create_patches(image_tensor)
        patch_list = [
            (self.extract_path(nii_path, i), patches_tensor[i].unsqueeze(0))
            for i in range(patches_tensor.size(0))
        ]
        return patch_list

    def extract_tensor(
        self, image_tensor: torch.Tensor, patch_index: int
    ) -> torch.Tensor:
        """Extracts a single patch from image_tensor"""
        patches_tensor = self.create_patches(image_tensor)
        return patches_tensor[patch_index, ...].unsqueeze_(0).clone()

    def extract_path(self, img_path: Path, patch_index: int) -> Path:
        """Constructs the save path for a given patch."""
        prefix_suffix = img_path.name.rsplit("_", 1)
        return Path(
            f"{prefix_suffix[0]}_patchsize-{self.patch_size}_stride-{self.stride_size}_patch-{patch_index}{prefix_suffix[1].replace(NII_GZ, PT)}"
        )

    def create_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Creates a tensor of patches from the image using `unfold`."""
        patches_tensor = (
            image_tensor.unfold(1, self.patch_size, self.stride_size)
            .unfold(2, self.patch_size, self.stride_size)
            .unfold(3, self.patch_size, self.stride_size)
            .contiguous()
        )
        return patches_tensor.view(
            -1, self.patch_size, self.patch_size, self.patch_size
        )
