from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from pydantic import field_validator, model_validator
from typing_extensions import Self

from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.exceptions import ClinicaDLArgumentError

from .base import BaseExtraction

logger = getLogger("clinicadl.preprocessing_config")

NII_GZ = ".nii.gz"
PT = ".pt"


class ROI(BaseExtraction):
    """
    Configuration class for extracting regions of interest (ROIs) from images using masks.
    """

    roi_list: List[str]
    roi_mask_location: Path
    roi_crop_input: bool = False
    roi_crop_output: bool = True
    roi_template: str = "MNI152NLin2009cSym"
    roi_mask_pattern: str = "res-1x1x1"

    # roi_custom_template: str = ""
    # roi_custom_mask_pattern: str = ""
    extract_method: ExtractionMethod = ExtractionMethod.ROI

    @field_validator("roi_mask_pattern", mode="before")
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
        """Check that list of ROIs is not empty"""
        if not v:
            raise NotImplementedError(
                "Default regions are not available anymore in ClinicaDL. "
                "Please define appropriate masks and give a roi_list."
            )
        if len(v) == 0:
            raise ClinicaDLArgumentError("A list of regions of interest must be given.")

        return v

    @field_validator("roi_mask_location", mode="before")
    def validate_roi_mask_location(cls, v: Path) -> Path:
        """Check if the given path is valid"""
        if not v:
            raise ClinicaDLArgumentError("A path to the mask location must be given.")
        if isinstance(v, str):
            v = Path(v)
        if not v.is_dir():
            raise ClinicaDLArgumentError(
                f"The path '{v}' is not a directory, please give another directory with masks location"
            )

        return v

    @model_validator(mode="after")
    def check_mask_list(self) -> Self:
        if self.roi_mask_location.resolve().parts[-1] != f"tpl-{self.roi_template}":
            self.roi_mask_location = (
                self.roi_mask_location / f"tpl-{self.roi_template}"
            )  # caps_directory / "masks" = mask_location

        for roi in self.roi_list:
            roi_path, desc = self.find_mask_path(roi)
            if roi_path is None:
                raise FileNotFoundError(
                    f"The ROI '{roi}' does not correspond to a mask in the CAPS directory. {desc}"
                )
            roi_mask = nib.loadsave.load(roi_path).get_fdata()  # type: ignore # do we need to check here ?
            mask_values = set(np.unique(roi_mask))
            if mask_values != {0, 1}:
                raise ValueError(
                    "The ROI masks used should be binary (composed of 0 and 1 only)."
                )
        return self

    def num_elem_per_image(self, image: torch.Tensor) -> int:
        return len(self.roi_list)

    # def check_preprocessing(self, preprocessing: Preprocessing):
    #     if preprocessing == Preprocessing.CUSTOM:
    #         if not self.roi_template:
    #             raise ClinicaDLArgumentError(
    #                 "A custom template must be defined when the modality is set to custom."
    #             )
    #         # self.roi_template = self.roi_custom_template
    #         # self.roi_mask_pattern = self.roi_custom_mask_pattern
    #     else:
    #         if preprocessing == Preprocessing.T1_LINEAR:
    #             self.roi_template = Template.T1_LINEAR
    #             self.roi_mask_pattern = Pattern.T1_LINEAR
    #         elif preprocessing == Preprocessing.PET_LINEAR:
    #             self.roi_template = Template.PET_LINEAR
    #             self.roi_mask_pattern = Pattern.PET_LINEAR
    #         elif preprocessing == Preprocessing.FLAIR_LINEAR:
    #             self.roi_template = Template.FLAIR_LINEAR
    #             self.roi_mask_pattern = Pattern.FLAIR_LINEAR

    def find_mask_path(self, roi: str) -> Tuple[Path, str]:
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

        candidates_pattern = f"*{self.roi_mask_pattern}*roi-{roi}_mask.nii*"

        desc = f"The mask should follow the pattern {candidates_pattern}. "

        candidates = [e for e in self.roi_mask_location.glob(candidates_pattern)]

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
            mask_path, _ = self.find_mask_path(roi_name)
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

        mask_paths, mask_arrays = list(), list()
        for roi in self.roi_list:
            logger.info(f"Find mask for roi {roi}.")
            mask_path, desc = self.find_mask_path(roi)
            if mask_path is None:
                raise FileNotFoundError(desc)
            mask_nii = nib.loadsave.load(mask_path)
            mask_paths.append(Path(mask_path))
            mask_arrays.append(mask_nii.get_fdata())  # type: ignore

        return mask_paths, mask_arrays
