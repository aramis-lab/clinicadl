from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import torch
from pydantic import PositiveInt, computed_field, field_validator

from clinicadl.transforms.extraction.base import Extraction
from clinicadl.utils.enum import ExtractionMethod

logger = getLogger("clinicadl.extraction.patch")

PT = ".pt"


class Patch(Extraction):
    """
    Transform class to extract patches from an image.

    This class enables patches extraction from an image tensor. The image is divided into smaller patches
    using a sliding window approach, where the patch size and the stride are configurable.

    Parameters
    ----------
    patch_size :  Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] (optional, default=50)
        The size of each patch. If a single value is passed, the same patch size will be used for the three
        spatial dimensions.
    stride : Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] (optional, default=50)
        The stride or step size used to move the sliding window. If a single value is passed, the same patch
        stride will be used for the three spatial dimensions.
    """

    patch_size: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] = 50
    stride: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]] = 50

    @computed_field
    @property
    def extract_method(self) -> ExtractionMethod:
        """The method to be used for the extraction process (ROI, Image, Patch, Slice)."""
        return ExtractionMethod.PATCH

    @field_validator("patch_size", "stride", mode="after")
    @classmethod
    def ensure_tuples(
        cls, v: Union[PositiveInt, Tuple[PositiveInt, PositiveInt, PositiveInt]]
    ) -> Tuple[PositiveInt, PositiveInt, PositiveInt]:
        """
        Ensures that 'patch_size' and 'stride' is always a tuple.
        """
        if isinstance(v, int):
            return (v, v, v)
        else:
            return v

    def num_sample_per_image(self, image: torch.Tensor) -> int:
        """
        Returns the total number of patches extracted from an image.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor (4D), where the first dimension represents the channel dimension.

        Returns
        -------
        int
            The total number of patches that can be extracted from the image.

        Notes
        -----
        The number of patches is determined by the image size, the patch size, and the stride.
        """
        return self.get_patches(image).shape[0]

    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Extracts all the patches from a NIfTI image tensor.

        Parameters
        ----------
        nii_path : Path
            The path to the NIfTI image file from which patches will be extracted.

        Returns
        -------
        List[Tuple[Path, torch.Tensor]]
            A list of tuples, where each tuple contains an extracted patch,
            and the path where to store it.

        Notes
        -----
        This method extracts patches from the full image and saves them as separate tensors.
        Each patch tensor is returned along with its associated file path.
        """

        image_tensor = self.load_image(nii_path)
        patches_tensor = self.get_patches(image_tensor)
        patch_list = [
            (self.sample_path(nii_path, idx), patches_tensor[i])
            for idx in range(patches_tensor.size(0))
        ]
        return patch_list

    def extract_sample(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> torch.Tensor:
        """
        Extracts a single patch from an image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor from which a patch will be extracted. Must be a 4D tensor
            with a channel dimension and 3 spatial dimensions.
        patch_index : int
            The index of the patch to extract from the image tensor.

        Returns
        -------
        torch.Tensor
            The extracted patch as a 4D tensor (with a channel dimension).

        Notes
        -----
        This method allows for the extraction of individual patches based on the provided index.
        """
        patches_tensor = self.get_patches(image_tensor)
        return patches_tensor[sample_index, ...].unsqueeze_(0).clone()

    def sample_path(self, image_path: Path, sample_index: int) -> Path:
        """
        Constructs the path to save a given patch.

        Parameters
        ----------
        image_path : Path
            The original image path, used to derive the path for saving the patch.
        sample_index : int
            The index of the patch being saved.

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
        prefix_suffix = image_path.name.rsplit("_", 1)
        patch_size_str = "x".join([str(s) for s in self.patch_size])
        stride_str = "x".join([str(s) for s in self.patch_size])
        return (
            Path(
                f"{prefix_suffix[0]}_patchsize-{patch_size_str}_stride-{stride_str}_patch-{sample_index}_{prefix_suffix[1]}"
            )
            .with_suffix("")
            .with_suffix(PT)
        )

    def get_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Creates a tensor of patches from the image using the PyTorch method `unfold`.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor (4D), where the first dimension represents the channel dimension.

        Returns
        -------
        torch.Tensor
            A tensor containing all the patches extracted from the image. The tensor shape
            will be `(num_patches, patch_size[0], patch_size[1], patch_size[2])`, where `num_patches` is
            determined by the image size, the patch size, and the stride.

        Notes
        -----
        The `unfold` method is used to extract patches by sliding a window over the image.
        The patches are then reshaped into a 4D tensor where each patch is a separate element.
        """
        patches_tensor = (
            image_tensor.unfold(1, self.patch_size[0], self.stride[0])
            .unfold(2, self.patch_size[1], self.stride[1])
            .unfold(3, self.patch_size[2], self.stride[2])
            .contiguous()
        )
        return patches_tensor.view(
            -1, self.patch_size, self.patch_size, self.patch_size
        )
