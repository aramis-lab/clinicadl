from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import torch
from pydantic import PositiveInt

from clinicadl.transforms.extraction.base import BaseExtraction
from clinicadl.utils.enum import ExtractionMethod

logger = getLogger("clinicadl.extraction.patch")

NII_GZ = ".nii.gz"
PT = ".pt"


class Patch(BaseExtraction):
    """
    Configuration class for patch extraction from an image with defined patch size and stride.

    This class extracts patches from an image tensor. The image is divided into smaller patches
    using a sliding window approach, where the patch size and stride size are configurable.

    Attributes
    ----------
    patch_size : int
        The size of each patch (default is 50).
    stride_size : int
        The stride or step size used to move the sliding window (default is 50).
    extract_method : ExtractionMethod
        The extraction method used for this class, set to PATCH.
    """

    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH

    def num_elem_per_image(self, image: torch.Tensor) -> PositiveInt:
        """
        Returns the total number of patches generated from the image.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor from which patches will be created.

        Returns
        -------
        int
            The total number of patches that can be extracted from the image.

        Notes
        -----
        The number of patches is determined by the image size, patch size, and stride size.
        """
        return self.create_patches(image).shape[0]

    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Extracts patches from a NIfTI image tensor.

        Parameters
        ----------
        nii_path : Path
            The path to the NIfTI image file from which patches will be extracted.

        Returns
        -------
        List[Tuple[Path, torch.Tensor]]
            A list of tuples where each tuple contains the path to save the patch
            and the corresponding patch tensor.

        Notes
        -----
        This method extracts patches from the full image and saves them as separate tensors.
        Each patch tensor is returned along with its associated file path.
        """

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
        """
        Extracts a single patch from the image tensor.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor from which a patch will be extracted.
        patch_index : int
            The index of the patch to extract from the image tensor.

        Returns
        -------
        torch.Tensor
            The extracted patch as a tensor, with a batch dimension added.

        Notes
        -----
        This method allows for the extraction of individual patches based on the provided index.
        """
        patches_tensor = self.create_patches(image_tensor)
        return patches_tensor[patch_index, ...].unsqueeze_(0).clone()

    def extract_path(self, img_path: Path, patch_index: int) -> Path:
        """
        Constructs the save path for a given patch.

        Parameters
        ----------
        img_path : Path
            The original image path used to derive the patch's save location.
        patch_index : int
            The index of the patch used to generate a unique filename.

        Returns
        -------
        Path
            The path where the patch will be saved, including the patch size and stride
            information in the filename.

        Notes
        -----
        The filename is generated using the original image name, appending patch size, stride,
        and the patch index to ensure each patch is saved with a unique name.
        """
        prefix_suffix = img_path.name.rsplit("_", 1)
        return Path(
            f"{prefix_suffix[0]}_patchsize-{self.patch_size}_stride-{self.stride_size}_patch-{patch_index}{prefix_suffix[1].replace(NII_GZ, PT)}"
        )

    def create_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Creates a tensor of patches from the image using `unfold`.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor from which patches will be extracted.

        Returns
        -------
        torch.Tensor
            A tensor containing all the patches extracted from the image. The tensor shape
            will be (num_patches, patch_size, patch_size, patch_size), where `num_patches` is
            determined by the image size, patch size, and stride.

        Notes
        -----
        The `unfold` method is used to extract patches by sliding a window over the image.
        The patches are then reshaped into a 4D tensor where each patch is a separate element.
        """
        patches_tensor = (
            image_tensor.unfold(1, self.patch_size, self.stride_size)
            .unfold(2, self.patch_size, self.stride_size)
            .unfold(3, self.patch_size, self.stride_size)
            .contiguous()
        )
        return patches_tensor.view(
            -1, self.patch_size, self.patch_size, self.patch_size
        )
