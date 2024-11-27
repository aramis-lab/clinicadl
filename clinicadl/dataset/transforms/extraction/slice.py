from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import torch
from pydantic import field_validator

from clinicadl.utils.enum import (
    ExtractionMethod,
    SliceDirection,
    SliceMode,
)

from .base import BaseExtraction

logger = getLogger("clinicadl.preprocessing_config")

NII_GZ = ".nii.gz"
PT = ".pt"


class Slice(BaseExtraction):
    """
    Configuration class for slice extraction from an image in specified directions.
    """

    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    slice_mode: SliceMode = SliceMode.RGB
    # num_slices: Optional[NonNegativeInt] = None # not sure it is needed
    discarded_slices: Tuple[int, int] = (0, 0)
    extract_method: ExtractionMethod = ExtractionMethod.SLICE

    @field_validator("discarded_slices", mode="before")
    def validate_discarded_slice(cls, v: Union[int, Tuple]) -> Tuple[int, int]:
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

    def num_elem_per_image(self, image: torch.Tensor):
        direction = int(self.slice_direction)
        return image.size(direction + 1) - sum(self.discarded_slices)

    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """Extracts slices from the image in the specified direction."""
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
        """Constructs the save path for a given slice."""
        prefix_suffix = img_path.name.rsplit("_", 1)
        slice_dict = {0: "sag", 1: "cor", 2: "axi"}

        return Path(
            f"{prefix_suffix[0]}_axis-{slice_dict[int(self.slice_direction.value)]}"
            f"_channel-{self.slice_mode.value}_slice-{slice_index}{prefix_suffix[1].replace(NII_GZ, PT)}"
        )
