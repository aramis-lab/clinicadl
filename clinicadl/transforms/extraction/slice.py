from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import torch
from pydantic import field_validator

from clinicadl.transforms.extraction.base import BaseExtraction
from clinicadl.utils.enum import (
    ExtractionMethod,
    SliceDirection,
    SliceMode,
)

logger = getLogger("clinicadl.extraction.slice")

NII_GZ = ".nii.gz"
PT = ".pt"


class Slice(BaseExtraction):
    """
    Configuration class for slice extraction from an image in specified directions.

    This class allows users to define extraction configurations for obtaining slices from a 3D image tensor.
    The extracted slices can be processed in different directions (e.g., sagittal, coronal, axial) and can
    be adjusted for RGB mode.

    """

    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    slice_mode: SliceMode = SliceMode.RGB
    discarded_slices: Tuple[int, int] = (0, 0)
    extract_method: ExtractionMethod = ExtractionMethod.SLICE

    @field_validator("discarded_slices", mode="before")
    def validate_discarded_slice(cls, v: Union[int, Tuple]) -> Tuple[int, int]:
        """
        Validates the discarded_slices attribute, ensuring it is either a single integer or a tuple of two integers.

        Raises
        ------
        IndexError
            If the value for discarded_slices is neither an integer nor a tuple with one or two elements.
        """
        if isinstance(v, int):
            return (v, v)
        elif len(v) == 1:
            return (v[0], v[0])
        elif len(v) == 2:
            return v
        else:
            raise IndexError(
                f"Maximum two number of discarded slices can be defined. "
                f"You gave discarded slices = {v}."
            )

    def num_elem_per_image(self, image: torch.Tensor) -> int:
        """
        Returns the number of slices that can be extracted from the input image tensor,
        accounting for the discarded slices at the start and end.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor (4D), where the first dimension represents the batch size
            and the second dimension represents the slices in the specified direction.

        Returns
        -------
        int
            The number of slices available after applying the discarded slices.
        """
        direction = int(self.slice_direction)
        return image.size(direction + 1) - sum(self.discarded_slices)

    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Extracts slices from the image at the specified direction, accounting for the discarded slices.

        Parameters
        ----------
        nii_path : Path
            The path to the input NIfTI image file.

        Returns
        -------
        List[Tuple[Path, torch.Tensor]]
            A list of tuples, where each tuple contains the file path for saving the slice
            and the extracted slice tensor.
        """
        image_tensor = self.extract_image(nii_path)
        start, end = self.discarded_slices
        slices = []
        for i in range(
            start, image_tensor.size(int(self.slice_direction.value) + 1) - end
        ):
            slice_tensor = self.extract_tensor(image_tensor, i)
            slices.append((self.extract_path(nii_path, i), slice_tensor))
        return slices

    def extract_tensor(
        self, image_tensor: torch.Tensor, slice_index: int
    ) -> torch.Tensor:
        """
        Extracts a single slice from the image tensor at the specified index.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor, which is a 4D tensor with dimensions (batch_size, slices, height, width).
        slice_index : int
            The index of the slice to extract in the specified direction.

        Returns
        -------
        torch.Tensor
            A tensor representing the extracted slice, with dimensions (3, height, width) if in RGB mode,
            or (1, height, width) otherwise.
        """
        idx_tuple = tuple(
            [slice(None)] * (int(self.slice_direction) + 1)
            + [slice_index + self.discarded_slices[0]]
            + [slice(None)] * (2 - int(self.slice_direction))
        )
        slice_tensor = image_tensor[idx_tuple]  # shape is 1 * W * L
        if self.slice_mode == SliceMode.RGB:
            slice_tensor = torch.cat([slice_tensor] * 3)  # shape is 3 * W * L
        return slice_tensor.clone()

    def extract_path(self, img_path: Path, slice_index: int) -> Path:
        """
        Constructs the file path for saving a given slice, based on the input image path and slice index.

        Parameters
        ----------
        img_path : Path
            The path to the input image file.
        slice_index : int
            The index of the slice being saved.

        Returns
        -------
        Path
            The constructed file path for the slice.
        """
        prefix_suffix = img_path.name.rsplit("_", 1)
        slice_dict = {0: "sag", 1: "cor", 2: "axi"}

        return Path(
            f"{prefix_suffix[0]}_axis-{slice_dict[int(self.slice_direction.value)]}"
            f"_channel-{self.slice_mode.value}_slice-{slice_index}{prefix_suffix[1].replace(NII_GZ, PT)}"
        )
