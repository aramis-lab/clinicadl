from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from time import time
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeInt

from clinicadl.utils.enum import (
    ExtractionMethod,
    Pattern,
    Preprocessing,
    SliceDirection,
    SliceMode,
    Template,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing_config")

NII_GZ = ".nii.gz"
PT = ".pt"


class ExtractionConfig(BaseModel):
    """
    Abstract config class for the Extraction procedure.
    """

    extract_method: ExtractionMethod
    extract_json: str = f"extract_{int(time())}.json"
    use_uncropped_image: bool = True
    save_features: bool = False

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("extract_json", mode="before")
    def compute_extract_json(cls, v: str):
        """Ensures the extract_json filename has a .json extension."""
        if isinstance(v, Path):
            v = str(v)
        elif not v.endswith(".json"):
            v = f"{v}.json"
        return v

    def extract_image(self, input_img: Path) -> torch.Tensor:
        """Loads a NIfTI image and returns it as a float32 tensor."""
        image_array = nib.loadsave.load(input_img).get_fdata(dtype="float32")  # type: ignore
        return torch.from_numpy(image_array).unsqueeze(0).float()

    @abstractmethod
    def extract_tensor(
        self,
        image_tensor: torch.Tensor,
        index: int,
        object_tensors: Optional[torch.Tensor] = None,
    ):
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
    def num_elem_per_image(self, image: torch.Tensor, elem_index: Optional[int] = None):
        """Returns the number of extracted elements per image."""
        pass


class ExtractionImageConfig(ExtractionConfig):
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
    ):
        return image_tensor

    def extract_path(self, image_path, index):
        return image_path

    def num_elem_per_image(self, image: torch.Tensor, elem_index: Optional[int] = None):
        return 1


class ExtractionPatchConfig(ExtractionConfig):
    """
    Configuration class for patch extraction from an image with defined patch size and stride.
    """

    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH

    def num_elem_per_image(self, image: torch.Tensor, elem_index: Optional[int] = None):
        """Returns the total number of patches generated from the image."""
        if elem_index is not None:
            return 1

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


class ExtractionSliceConfig(ExtractionConfig):
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

    def num_elem_per_image(self, image: torch.Tensor, elem_index: Optional[int] = None):
        if elem_index is not None:
            return 1
        # if self.num_slices is not None:
        #     return self.num_slices
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


class ExtractionROIConfig(ExtractionConfig):
    """
    Configuration class for extracting regions of interest (ROIs) from images using masks.
    """

    roi_list: List[str] = []
    roi_crop_input: bool = True
    roi_crop_output: bool = True
    roi_template: str = ""
    roi_mask_pattern: str = ""
    roi_mask_location: Path

    # roi_custom_template: str = ""
    # roi_custom_mask_pattern: str = ""
    extract_method: ExtractionMethod = ExtractionMethod.ROI

    @field_validator("roi_mask_pattern", "before")
    def validate_roi_mask_pattern(cls, v: str) -> str:
        """Check that pattern begins and ends with _ to avoid mixing keys"""
        if not v:
            raise ClinicaDLArgumentError("A mask pattern must be defined.")
        if not v.startswith("_"):
            v = "_" + v
        if not v.endswith("_"):
            v = v + "_"
        return v

    @field_validator("roi_list", mode="before")
    def validate_roi_list(cls, v: List[str]) -> List[str]:
        if not v:
            raise NotImplementedError(
                "Default regions are not available anymore in ClinicaDL. "
                "Please define appropriate masks and give a roi_list."
            )
        if len(v) == 0:
            raise ClinicaDLArgumentError("A list of regions of interest must be given.")

        return v

    def num_elem_per_image(
        self, image: torch.Tensor, elem_index: Optional[int] = None
    ) -> int:
        return 1 if elem_index is not None else len(self.roi_list)

    def check_with_preprocessing(self, preprocessing: Preprocessing):
        if preprocessing == Preprocessing.CUSTOM:
            if not self.roi_template:
                raise ClinicaDLArgumentError(
                    "A custom template must be defined when the modality is set to custom."
                )
            # self.roi_template = self.roi_custom_template
            # self.roi_mask_pattern = self.roi_custom_mask_pattern
        else:
            if preprocessing == Preprocessing.T1_LINEAR:
                self.roi_template = Template.T1_LINEAR
                self.roi_mask_pattern = Pattern.T1_LINEAR
            elif preprocessing == Preprocessing.PET_LINEAR:
                self.roi_template = Template.PET_LINEAR
                self.roi_mask_pattern = Pattern.PET_LINEAR
            elif preprocessing == Preprocessing.FLAIR_LINEAR:
                self.roi_template = Template.FLAIR_LINEAR
                self.roi_mask_pattern = Pattern.FLAIR_LINEAR

    def check_mask_list(self, masks_location: Path) -> None:
        for roi in self.roi_list:
            roi_path, desc = self.find_mask_path(masks_location, roi)
            if roi_path is None:
                raise FileNotFoundError(
                    f"The ROI '{roi}' does not correspond to a mask in the CAPS directory. {desc}"
                )
            roi_mask = nib.loadsave.load(roi_path).get_fdata()  # type: ignore
            mask_values = set(np.unique(roi_mask))
            if mask_values != {0, 1}:
                raise ValueError(
                    "The ROI masks used should be binary (composed of 0 and 1 only)."
                )

    def find_mask_path(self, masks_location: Path, roi: str) -> Tuple[Path, str]:
        """
        Finds masks corresponding to the pattern asked and containing the adequate self.roi_crop_input description

        Parameters
        ----------
        masks_location: Path
            Directory containing the masks.
        roi: str
            Name of the region.

        Returns
        -------
            path of the mask or None if nothing was found.
            a human-friendly description of the pattern looked for.
        """

        candidates_pattern = f"*{self.roi_mask_pattern}*_roi-{roi}_mask.nii*"

        desc = f"The mask should follow the pattern {candidates_pattern}. "
        candidates = [e for e in masks_location.glob(candidates_pattern)]
        if self.roi_crop_input is None:
            # pass
            candidates2 = candidates
        elif self.roi_crop_input:
            candidates2 = [mask for mask in candidates if "_desc-Crop_" in mask.name]
            desc += "and contain '_desc-Crop_' string."
        else:
            candidates2 = [
                mask for mask in candidates if "_desc-Crop_" not in mask.name
            ]
            desc += "and not contain '_desc-Crop_' string."

        if len(candidates2) == 0:
            raise FileNotFoundError(
                f"Could not find any masks corresponding to the pattern asked and containing the adequate {self.roi_crop_input} description "
            )
            # return None, desc
        else:
            return min(candidates2), desc

    def compute_output_pattern(self, mask_path: Path):
        """
        Computes the output pattern of the region cropped (without the source file prefix)
        Parameters
        ----------
        mask_path: Path
            Path to the masks
        self.roi_crop_output: bool
            If True the output is cropped, and the descriptor CropRoi must exist

        Returns
        -------
            the output pattern
        """

        mask_filename = mask_path.name
        template_id = mask_filename.split("_")[0].split("-")[1]
        mask_descriptors = mask_filename.split("_")[1:-2:]
        roi_id = mask_filename.split("_")[-2].split("-")[1]
        if "desc-Crop" not in mask_descriptors and not self.roi_crop_output:
            mask_descriptors = ["desc-CropRoi"] + mask_descriptors
        elif "desc-Crop" in mask_descriptors:
            mask_descriptors = [
                descriptor
                for descriptor in mask_descriptors
                if descriptor != "desc-Crop"
            ]
            if self.roi_crop_output:
                mask_descriptors = ["desc-CropRoi"] + mask_descriptors
            else:
                mask_descriptors = ["desc-CropImage"] + mask_descriptors

        mask_pattern = "_".join(mask_descriptors)

        if mask_pattern == "":
            output_pattern = f"space-{template_id}_roi-{roi_id}"
        else:
            output_pattern = f"space-{template_id}_{mask_pattern}_roi-{roi_id}"

        return output_pattern

    def extract(self, nii_path: Path) -> List[Tuple[str, torch.Tensor]]:
        """Extracts roi from a NIfTI image tensor."""
        image_tensor = self.extract_image(nii_path)
        roi_list = []
        for roi_name in self.roi_list:
            mask_path, _ = self.find_mask_path(self.roi_mask_location, roi_name)
            mask_np = nib.loadsave.load(mask_path).get_fdata()  # type: ignore
            roi_list.append(
                (
                    self.extract_tensor(image_tensor, mask_np),
                    self.extract_path(nii_path, mask_path),
                )
            )
        return roi_list

    def extract_tensor(self, image_tensor: torch.Tensor, roi_idx: int) -> torch.Tensor:
        _, mask_arrays = self._get_mask_paths_and_tensors()
        mask_np = mask_arrays[roi_idx]

        if len(mask_np.shape) == 3:
            mask_np = np.expand_dims(mask_np, axis=0)
        elif len(mask_np.shape) == 4:
            assert mask_np.shape[0] == 1
        else:
            raise ValueError(
                "ROI masks must be 3D or 4D tensors. "
                f"The dimension of your ROI mask is {len(mask_np.shape)}."
            )

        roi_tensor = image_tensor * mask_np
        if self.roi_crop_output:
            roi_tensor = roi_tensor[
                np.ix_(
                    mask_np.any((1, 2, 3)),
                    mask_np.any((0, 2, 3)),
                    mask_np.any((0, 1, 3)),
                    mask_np.any((0, 1, 2)),
                )
            ]
        return roi_tensor.float().clone()

    def extract_path(self, img_path: Path, mask_path: Path) -> str:
        input_img_filename = img_path.name

        sub_ses_prefix = "_".join(input_img_filename.split("_")[0:3:])
        if not sub_ses_prefix.endswith("_T1w"):
            sub_ses_prefix = "_".join(input_img_filename.split("_")[0:2:])
        input_suffix = input_img_filename.split("_")[-1].split(".")[0]

        output_pattern = self.compute_output_pattern(mask_path)

        return f"{sub_ses_prefix}_{output_pattern}_{input_suffix}{PT}"

    def _get_mask_paths_and_tensors(self) -> Tuple[List[str], List]:
        """Loads the masks necessary to regions extraction"""
        mask_location = (
            self.roi_mask_location
            / f"tpl-{self.roi_template}"  # caps_directory / "masks" = mask_location
        )

        mask_paths, mask_arrays = list(), list()
        for roi in self.roi_list:
            logger.info(f"Find mask for roi {roi}.")
            mask_path, desc = self.find_mask_path(mask_location, roi)
            if mask_path is None:
                raise FileNotFoundError(desc)
            mask_nii = nib.loadsave.load(mask_path)
            mask_paths.append(Path(mask_path))
            mask_arrays.append(mask_nii.get_fdata())  # type: ignore

        return mask_paths, mask_arrays


ALL_EXTRACTION_TYPES = Union[
    ExtractionImageConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
    ExtractionPatchConfig,
]
