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
    Suffix,
    Template,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing_config")


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
        if isinstance(v, Path):
            v = str(v)
        elif not v.endswith(".json"):
            v = f"{v}.json"
        return v


class ExtractionImageConfig(ExtractionConfig):
    extract_method: ExtractionMethod = ExtractionMethod.IMAGE

    def extract_images(self, input_img: Path) -> list[Tuple[Path, torch.Tensor]]:
        """Extract the images
        This function convert nifti image to tensor (.pt) version of the image.
        Tensor version is saved at the same location than input_img.
        Args:
            input_img: path to the NifTi input image.
        Returns:
            filename (str): single tensor file  saved on the disk. Same location than input file.
        """

        image_array = nib.loadsave.load(input_img).get_fdata(dtype="float32")
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
        # make sure the tensor type is torch.float32
        output_file = (
            Path(input_img.name.replace(Suffix.NII_GZ.value, Suffix.PT.value)),
            image_tensor.clone(),
        )

        return [output_file]


class ExtractionPatchConfig(ExtractionConfig):
    patch_size: int = 50
    stride_size: int = 50
    extract_method: ExtractionMethod = ExtractionMethod.PATCH

    def num_elem_per_image(self, image: torch.Tensor):
        # if self.elem_index is not None:
        #     return 1
        patches_tensor = self.create_patches(image)
        self.num_patches = patches_tensor.shape[0]
        return self.num_patches

    def extract_patches(
        self,
        nii_path: Path,
    ) -> List[Tuple[Path, torch.Tensor]]:
        """Extracts the patches
        This function extracts patches form the preprocessed nifti image. Patch size
        if provided as input and also the stride size. If stride size is smaller
        than the patch size an overlap exist between consecutive patches. If stride
        size is equal to path size there is no overlap. Otherwise, unprocessed
        zones can exits.
        Args:
            nii_path: path to the NifTi input image.
            self.patch_size: size of a single patch.
            self.stride_size: size of the stride leading to next patch.
        Returns:
            list of tuples containing the path to the extracted patch
                and the tensor of the corresponding patch.
        """

        image_array = nib.loadsave.load(nii_path).get_fdata(dtype="float32")
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

        patches_tensor = self.create_patches(image_tensor)

        patch_list = []
        for patch_index in range(patches_tensor.shape[0]):
            patch_tensor = self.extract_patch_tensor(
                image_tensor, patch_index, patches_tensor
            )
            patch_path = self.extract_patch_path(nii_path, patch_index)

            patch_list.append((patch_path, patch_tensor))

        return patch_list

    def create_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        patches_tensor = (
            image_tensor.unfold(1, self.patch_size, self.stride_size)
            .unfold(2, self.patch_size, self.stride_size)
            .unfold(3, self.patch_size, self.stride_size)
            .contiguous()
        )
        # the dimension of patches_tensor is [1, patch_num1, patch_num2, patch_num3, self.patch_size1, self.patch_size2, self.patch_size3]
        return patches_tensor.view(
            -1, self.patch_size, self.patch_size, self.patch_size
        )

    def extract_patch_tensor(
        self,
        image_tensor: torch.Tensor,
        patch_index: int,
        patches_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extracts a single patch from image_tensor"""

        if patches_tensor is None:
            patches_tensor = self.create_patches(image_tensor)

        return patches_tensor[patch_index, ...].unsqueeze_(0).clone()

    def extract_patch_path(self, img_path: Path, patch_index: int) -> Path:
        input_img_filename = img_path.name
        txt_idx = input_img_filename.rfind("_")
        it_filename_prefix = input_img_filename[0:txt_idx]
        it_filename_suffix = input_img_filename[txt_idx:]
        it_filename_suffix = it_filename_suffix.replace(
            Suffix.NII_GZ.value, Suffix.PT.value
        )

        return Path(
            f"{it_filename_prefix}_patchsize-{self.patch_size}_stride-{self.stride_size}_patch-{patch_index}{it_filename_suffix}"
        )


class ExtractionSliceConfig(ExtractionConfig):
    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    slice_mode: SliceMode = SliceMode.RGB
    # num_slices: Optional[NonNegativeInt] = None # not sure it is needed
    discarded_slices: Union[int, Tuple] = (0,)
    extract_method: ExtractionMethod = ExtractionMethod.SLICE

    @field_validator("slice_direction", mode="before")
    def check_slice_direction(cls, v: str):
        if isinstance(v, int):
            return SliceDirection(str(v))

    # @field_validator("discarded_slices", mode="before")
    # def compute_discarded_slice(cls, v: Union[int, Tuple]) -> Tuple[int, int]:
    #     return compute_discarded_slices(v)
    # DONE in extraction

    def compute_discarded_slices(self) -> Tuple[int, int]:
        if isinstance(self.discarded_slices, int):
            begin_discard, end_discard = self.discarded_slices, self.discarded_slices
        elif len(self.discarded_slices) == 1:
            begin_discard, end_discard = (
                self.discarded_slices[0],
                self.discarded_slices[0],
            )
        elif len(self.discarded_slices) == 2:
            begin_discard, end_discard = (
                self.discarded_slices[0],
                self.discarded_slices[1],
            )
        else:
            raise IndexError(
                f"Maximum two number of discarded slices can be defined. "
                f"You gave discarded slices = {self.discarded_slices}."
            )
        return begin_discard, end_discard

    def extract_slices(
        self,
        nii_path: Path,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Extracts the slices from three directions
        This function extracts slices form the preprocessed nifti image.

        The direction of extraction can be defined either on sagittal direction (0),
        coronal direction (1) or axial direction (other).

        The output slices can be stored following two modes:
        single (1 channel) or rgb (3 channels, all the same).

        Args:
            nii_path: path to the NifTi input image.
            slice_direction: along which axis slices are extracted.
            slice_mode: 'single' or 'rgb'.
            discarded_slices: Number of slices to discard at the beginning and the end of the image.
                Will be a tuple of two integers if the number of slices to discard at the beginning
                and at the end differ.
        Returns:
            list of tuples containing the path to the extracted slice
                and the tensor of the corresponding slice.
        """

        image_array = nib.loadsave.load(nii_path).get_fdata(dtype="float32")
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

        begin_discard, end_discard = self.compute_discarded_slices()
        index_list = range(
            begin_discard,
            image_tensor.shape[int(self.slice_direction.value) + 1] - end_discard,
        )

        slice_list = []
        for slice_index in index_list:
            slice_tensor = self.extract_slice_tensor(image_tensor, slice_index)
            slice_path = self.extract_slice_path(nii_path, slice_index)

            slice_list.append((slice_path, slice_tensor))

        return slice_list

    def extract_slice_tensor(
        self,
        image_tensor: torch.Tensor,
        slice_index: int,
    ) -> torch.Tensor:
        # Allow to select the slice `slice_index` in dimension `slice_direction`
        idx_tuple = tuple(
            [slice(None)] * (int(self.slice_direction.value) + 1)
            + [slice_index]
            + [slice(None)] * (2 - int(self.slice_direction.value))
        )
        slice_tensor = image_tensor[idx_tuple]  # shape is 1 * W * L

        if self.slice_mode == "rgb":
            slice_tensor = torch.cat(
                (slice_tensor, slice_tensor, slice_tensor)
            )  # shape is 3 * W * L

        return slice_tensor.clone()

    def extract_slice_path(
        self,
        img_path: Path,
        slice_index: int,
    ) -> str:
        slice_dict = {0: "sag", 1: "cor", 2: "axi"}
        input_img_filename = img_path.name
        txt_idx = input_img_filename.rfind("_")
        it_filename_prefix = input_img_filename[0:txt_idx]
        it_filename_suffix = input_img_filename[txt_idx:]
        it_filename_suffix = it_filename_suffix.replace(
            Suffix.NII_GZ.value, Suffix.PT.value
        )
        return (
            f"{it_filename_prefix}_axis-{slice_dict[int(self.slice_direction.value)]}"
            f"_channel-{self.slice_mode.value}_slice-{slice_index}{it_filename_suffix}"
        )


class ExtractionROIConfig(ExtractionConfig):
    roi_list: List[str] = []
    roi_crop_input: bool = True
    roi_crop_output: bool = True
    roi_template: str = ""
    roi_pattern: str = ""
    roi_background_value: int = 0

    roi_custom_template: str = ""
    roi_custom_mask_pattern: str = ""
    roi_custom_suffix: str = ""
    extract_method: ExtractionMethod = ExtractionMethod.ROI

    def check_with_preprocessing(self, preprocessing: Preprocessing):
        if preprocessing == Preprocessing.CUSTOM:
            if not self.roi_template:
                raise ClinicaDLArgumentError(
                    "A custom template must be defined when the modality is set to custom."
                )
            self.roi_template = self.roi_custom_template
            self.roi_mask_pattern = self.roi_custom_mask_pattern
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

    def check_mask_list(
        self,
        masks_location: Path,
    ) -> None:
        if len(self.roi_list) == 0:
            raise ClinicaDLArgumentError("A list of regions of interest must be given.")

        for roi in self.roi_list:
            roi_path, desc = self.find_mask_path(masks_location, roi)
            if roi_path is None:
                raise FileNotFoundError(
                    f"The ROI '{roi}' does not correspond to a mask in the CAPS directory. {desc}"
                )
            roi_mask = nib.loadsave.load(roi_path).get_fdata()
            mask_values = set(np.unique(roi_mask))
            if mask_values != {0, 1}:
                raise ValueError(
                    "The ROI masks used should be binary (composed of 0 and 1 only)."
                )

    def find_mask_path(
        self,
        masks_location: Path,
        roi: str,
    ) -> Tuple[Union[None, Path], str]:
        """
        Finds masks corresponding to the pattern asked and containing the adequate self.roi_crop_input description

        Parameters
        ----------
        masks_location: Path
            Directory containing the masks.
        roi: str
            Name of the region.
        mask_pattern: str
            Pattern which should be found in the filename of the mask.
        self.roi_crop_input: bool
            If True the original image should contain the substring 'desc-Crop'.

        Returns
        -------
            path of the mask or None if nothing was found.
            a human-friendly description of the pattern looked for.
        """

        # Check that pattern begins and ends with _ to avoid mixing keys
        if self.roi_mask_pattern is None:
            mask_pattern = ""

        candidates_pattern = f"*{mask_pattern}*_roi-{roi}_mask.nii*"

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
            return None, desc
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

    def extract_roi(
        self,
        nii_path: Path,
        masks_location: Path,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Extracts regions of interest defined by masks
        This function extracts regions of interest from preprocessed nifti images.
        The regions are defined using binary masks that must be located in the CAPS
        at `masks/tpl-<template>`

        Parameters
        ----------
        nii_path: Path
            Path to the NifTi input image.
        masks_location: Path
            Path to the masks
        mask_pattern: str
            Pattern to identify the masks
        self.roi_crop_input: bool
            If the input is cropped or not (contains desc-Crop)
        roi_names: List[str]
            List of the names of the regions that will be extracted.
        unself.roi_crop_output: bool
            If True, the final region is not cropped.

        Returns
        -------
        list of tuples containing the path to the extracted ROI
            and the tensor of the corresponding ROI.
        """

        image_array = nib.loadsave.load(nii_path).get_fdata(dtype="float32")
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

        roi_list = []
        for roi_name in self.roi_list:
            # read mask
            mask_path, _ = self.find_mask_path(masks_location, roi_name)
            mask_np = nib.loadsave.load(mask_path).get_fdata()

            roi_tensor = self.extract_roi_tensor(image_tensor, mask_np)
            roi_path = self.extract_roi_path(nii_path, mask_path)

            roi_list.append((roi_path, roi_tensor))

        return roi_list

    def extract_roi_tensor(
        self,
        image_tensor: torch.Tensor,
        mask_np,
    ) -> torch.Tensor:
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

    def extract_roi_path(self, img_path: Path, mask_path: Path) -> str:
        input_img_filename = img_path.name

        sub_ses_prefix = "_".join(input_img_filename.split("_")[0:3:])
        if not sub_ses_prefix.endswith("_T1w"):
            sub_ses_prefix = "_".join(input_img_filename.split("_")[0:2:])
        input_suffix = input_img_filename.split("_")[-1].split(".")[0]

        output_pattern = self.compute_output_pattern(mask_path)

        return f"{sub_ses_prefix}_{output_pattern}_{input_suffix}.pt"


ALL_EXTRACTION_TYPES = Union[
    ExtractionImageConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
    ExtractionPatchConfig,
]
