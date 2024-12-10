from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from pydantic import (
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)
from typing_extensions import Self

from clinicadl.transforms.extraction.base import Extraction
from clinicadl.utils.enum import ExtractionMethod

logger = getLogger("clinicadl.extraction.roi")

PT = ".pt"


class ROI(Extraction):
    """
    Transform class to extract Regions of Interest (ROI) from an image.

    The ROIs are passed via binary images stored in NIfTI files. The transform will
    set all the voxels outside the ROI to 0, and can crop the output image the remove
    regions that are always set to 0 (i.e. the regions that are outside all masks).

    Parameters
    ----------
    masks : List[Path]
        The list of paths where to find the masks. All the masks must be 3D image of the same shape.
    crop : bool (optional, default=False)
        Whether to crop the extracted ROIs. If False, the ROI will be the same size as the input image (with
        possibly many zeros). If True, only the smallest interesting region will be kept. The smallest
        interesting region is defined as the union of the non-zero regions of all masks.
        The output size can be accessed via the attribute `output_size`.
    """

    masks: List[Path]
    crop: bool = False
    _mask_tensors: Optional[List[torch.Tensor]] = None
    _crop_indices: Optional[
        Tuple[
            NonNegativeInt,
            NonNegativeInt,
            NonNegativeInt,
            NonNegativeInt,
            NonNegativeInt,
            NonNegativeInt,
        ]
    ] = None
    _mask_size: Optional[Tuple[PositiveInt, PositiveInt, PositiveInt]] = None
    _output_size: Optional[Tuple[PositiveInt, PositiveInt, PositiveInt]] = None

    @computed_field
    @property
    def extract_method(self) -> ExtractionMethod:
        """The method to be used for the extraction process (ROI, Image, Patch, Slice)."""
        return ExtractionMethod.ROI

    @computed_field
    @property
    def output_size(self) -> Tuple[PositiveInt, PositiveInt, PositiveInt]:
        """
        The size of a sample. Equal to the size of the masks if 'crop' is False,
        otherwise it is equal to the size of the union of the non-zero regions
        of all masks.
        """
        return self._output_size

    @model_validator(mode="after")
    def validate_masks(self) -> Self:
        """Checks the path mask paths."""
        if len(self.masks) == 0:
            raise ValueError(
                "A list of regions of interest must be given. Got an empty list for 'masks'."
            )

        self._mask_tensors = []
        for path in self.masks:
            roi_mask = self.load_image(path).int()

            mask_values = set(roi_mask.unique().tolist())
            if mask_values != {0, 1}:
                raise ValueError(
                    f"The ROI masks should be binary (composed of 0 and 1 only). Got other values in {path}."
                )

            if len(roi_mask.shape) != 4:
                raise ValueError(
                    f"Expected 3D mask, but got {len(roi_mask.shape)-1} dimensions in {path}."
                )

            self._mask_tensors.append(roi_mask)

        self._mask_size = self._mask_tensors[0].shape[1:]
        for mask, path in zip(self._mask_tensors[1:], self.masks[1:]):
            shape = mask.shape[1:]
            if shape != self._mask_size:
                raise ValueError(
                    f"The masks must have the same shape. Got shape {shape} for {path} and "
                    f"shape {self._mask_size} for {self.masks[0]}"
                )

        if self.crop:
            merged_mask = torch.stack(
                self._mask_tensors,
            ).sum(0)
            self._crop_indices = self._crop_mask(merged_mask)
            self._output_size = (
                self._crop_indices[1] - self._crop_indices[0] + 1,
                self._crop_indices[3] - self._crop_indices[2] + 1,
                self._crop_indices[5] - self._crop_indices[4] + 1,
            )
        else:
            self._output_size = tuple(self._mask_tensors[0].shape[1:])

        return self

    @staticmethod
    def _crop_mask(
        mask: torch.Tensor,
    ) -> Tuple[
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
    ]:
        """Crops a mask to remove borders with only 0s."""
        _, x, y, z = torch.nonzero(mask, as_tuple=True)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        return (
            x_min.item(),
            x_max.item(),
            y_min.item(),
            y_max.item(),
            z_min.item(),
            z_max.item(),
        )

    def num_samples_per_image(self, image: torch.Tensor) -> int:
        """
        Returns the number of ROIs in an image.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor (4D), where the first dimension represents the channel dimension.

        Returns
        -------
        int
            The total number of ROIs in the image.

        Notes
        -----
        This method is a placeholder implemented for consistency with the parent class `Extraction`.
        """
        return len(self.masks)

    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Extracts all the regions of interest (ROIs) from a given NIfTI image.

        Parameters
        ----------
        nii_path : Path
            The path to the NIfTI image file.

        Returns
        -------
        List[Tuple[Path, torch.Tensor]]
            A list of tuples, where each tuple contains an extracted ROI,
            and the path where to store it.
        """
        image_tensor = self.load_image(nii_path)
        roi_list = [
            (self.sample_path(nii_path, idx), self.extract_sample(image_tensor, idx))
            for idx in range(self.num_samples_per_image(image_tensor))
        ]
        return roi_list

    def extract_sample(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> torch.Tensor:
        """
        Extracts a single ROI from an image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor from which a ROI will be extracted. Must be a 4D tensor
            with a channel dimension and 3 spatial dimensions.
        sample_index : int
            The index of the ROI to extract.

        Returns
        -------
        torch.Tensor
            The extracted patch as a 4D tensor (with a channel dimension). If 'crop' is False, the output
            is the same size as the input image. Otherwise, the output size is computed with the masks
            and can be accessed via the attribute 'output_size'.

        Raises
        ------
        ValueError
            If the image is not the same size as the mask.
        IndexError
            If 'sample_index' is greater or equal to the number of ROIs.
        """
        if tuple(image_tensor.shape) != (1, *self._mask_size):
            raise ValueError(
                f"The image must be the same size as the mask (i.e. {self._mask_size}). Got {tuple(image_tensor.shape)}"
            )
        if sample_index >= len(self.masks):
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there are only "
                f"{len(self.masks)} ROIs."
            )

        roi_tensor = image_tensor * self._mask_tensors[sample_index]
        if self.crop:
            x_min, x_max, y_min, y_max, z_min, z_max = self._crop_indices
            roi_tensor = roi_tensor[
                :, x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1
            ]

        return roi_tensor.float().clone()

    # TODO : to revise -> roi filename is not unique
    def sample_path(self, image_path: Path, sample_index: int) -> Path:
        """
        Computes the output path for the extracted ROI.

        Parameters
        ----------
        image_path : Path
            The original image path, used to derive the path for saving the ROI.
        sample_index : int
            The index of the ROI being saved.

        Returns
        -------
        Path
            The path where the ROI will be saved, including ROI information in the filename.
        """
        parent = image_path.parent
        prefix_suffix = image_path.name.rsplit("_", 1)
        roi_name = self.masks[sample_index].with_suffix("").with_suffix("").name
        return (
            (parent / f"{prefix_suffix[0]}_roi-{roi_name}_{prefix_suffix[1]}")
            .with_suffix("")
            .with_suffix(PT)
        )
