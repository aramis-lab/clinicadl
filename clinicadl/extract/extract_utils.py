# coding: utf8
from os import path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def get_parameters_dict(
    modality,
    extract_method,
    save_features,
    use_uncropped_image,
    custom_suffix,
    acq_label,
    suvr_reference_region,
):
    parameters = {
        "preprocessing": modality,
        "mode": extract_method,
        "use_uncropped_image": use_uncropped_image,
        "prepare_dl": save_features,
    }

    if modality == "custom":
        parameters["custom_suffix"] = custom_suffix
    if modality == "pet-linear":
        parameters["acq_label"] = acq_label
        parameters["suvr_reference_region"] = suvr_reference_region

    return parameters



############
# SLICE    #
############
def extract_slices(
    nii_path: str,
    slice_direction: int = 0,
    slice_mode: str = "single",
    discarded_slices: Union[int, tuple] = 0,
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
    import nibabel as nib

    image_array = nib.load(nii_path).get_fdata(dtype="float32")
    image_tensor = torch.from_numpy(image_array).float()

    # Remove discarded slices
    if isinstance(discarded_slices, int):
        begin_discard, end_discard = discarded_slices, discarded_slices
    elif len(discarded_slices) == 1:
        begin_discard, end_discard = discarded_slices[0], discarded_slices[0]
    elif len(discarded_slices) == 2:
        begin_discard, end_discard = discarded_slices[0], discarded_slices[1]
    else:
        raise ValueError(
            f"Maximum two number of discarded slices can be defined. "
            f"You gave discarded slices = {discarded_slices}."
        )
    index_list = range(begin_discard, image_tensor.shape[slice_direction] - end_discard)

    slice_list = []
    for slice_index in index_list:
        slice_tensor = extract_slice_tensor(
            image_tensor, slice_direction, slice_mode, slice_index
        )
        slice_path = extract_slice_path(
            nii_path, slice_direction, slice_mode, slice_index
        )

        slice_list.append((slice_path, slice_tensor))

    return slice_list


def extract_slice_tensor(
    image_tensor: torch.Tensor,
    slice_direction: int,
    slice_mode: str,
    slice_index: int,
) -> torch.Tensor:
    # Allow to select the slice `slice_index` in dimension `slice_direction`
    idx_tuple = tuple(
        [slice(None)] * slice_direction
        + [slice_index]
        + [slice(None)] * (2 - slice_direction)
    )
    slice_tensor = image_tensor[idx_tuple]
    slice_tensor.unsqueeze_(0)  # shape is 1 * W * L

    if slice_mode == "rgb":
        slice_tensor = torch.cat(
            (slice_tensor, slice_tensor, slice_tensor)
        )  # shape is 3 * W * L

    return slice_tensor.clone()


def extract_slice_path(
    img_path: str, slice_direction: int, slice_mode: str, slice_index: int
) -> str:

    direction_dict = {0: "sag", 1: "cor", 2: "axi"}
    if slice_direction not in direction_dict:
        raise ValueError(
            f"Slice direction {slice_direction} should be in the keys of {direction_dict}."
        )

    input_img_filename = path.basename(img_path)
    txt_idx = input_img_filename.rfind("_")
    it_filename_prefix = input_img_filename[0:txt_idx]
    it_filename_suffix = input_img_filename[txt_idx:]
    it_filename_suffix = it_filename_suffix.replace(".nii.gz", ".pt")
    return (
        f"{it_filename_prefix}_axis-{direction_dict[slice_direction]}"
        f"_channel-{slice_mode}_slice-{slice_index}{it_filename_suffix}"
    )


############
# PATCH    #
############
def extract_patches(
    nii_path: str,
    patch_size: int,
    stride_size: int,
) -> List[Tuple[str, torch.Tensor]]:
    """Extracts the patches
    This function extracts patches form the preprocessed nifti image. Patch size
    if provided as input and also the stride size. If stride size is smaller
    than the patch size an overlap exist between consecutive patches. If stride
    size is equal to path size there is no overlap. Otherwise, unprocessed
    zones can exits.
    Args:
        nii_path: path to the NifTi input image.
        patch_size: size of a single patch.
        stride_size: size of the stride leading to next patch.
    Returns:
        list of tuples containing the path to the extracted patch
            and the tensor of the corresponding patch.
    """
    import nibabel as nib

    image_array = nib.load(nii_path).get_fdata(dtype="float32")
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

    patches_tensor = (
        image_tensor.unfold(1, patch_size, stride_size)
        .unfold(2, patch_size, stride_size)
        .unfold(3, patch_size, stride_size)
        .contiguous()
    )
    patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)

    patch_list = []
    for patch_index in range(patches_tensor.shape[0]):
        patch_tensor = extract_patch_tensor(
            image_tensor, patch_size, stride_size, patch_index, patches_tensor
        )
        patch_path = extract_patch_path(nii_path, patch_size, stride_size, patch_index)

        patch_list.append((patch_path, patch_tensor))

    return patch_list


def extract_patch_tensor(
    image_tensor: torch.Tensor,
    patch_size: int,
    stride_size: int,
    patch_index: int,
    patches_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """Extracts a single patch from image_tensor"""

    if patches_tensor is None:
        patches_tensor = (
            image_tensor.unfold(1, patch_size, stride_size)
            .unfold(2, patch_size, stride_size)
            .unfold(3, patch_size, stride_size)
            .contiguous()
        )

        # the dimension of patches_tensor is [1, patch_num1, patch_num2, patch_num3, patch_size1, patch_size2, patch_size3]
        patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)

    return patches_tensor[patch_index, ...].unsqueeze_(0).clone()


def extract_patch_path(
    img_path: str, patch_size: int, stride_size: int, patch_index: int
) -> str:
    input_img_filename = path.basename(img_path)
    txt_idx = input_img_filename.rfind("_")
    it_filename_prefix = input_img_filename[0:txt_idx]
    it_filename_suffix = input_img_filename[txt_idx:]
    it_filename_suffix = it_filename_suffix.replace(".nii.gz", ".pt")

    return f"{it_filename_prefix}_patchsize-{patch_size}_stride-{stride_size}_patch-{patch_index}{it_filename_suffix}"


############
# IMAGE    #
############
def extract_images(input_img: str) -> List[Tuple[str, torch.Tensor]]:
    """Extract the images
    This function convert nifti image to tensor (.pt) version of the image.
    Tensor version is saved at the same location than input_img.
    Args:
        input_img: path to the NifTi input image.
    Returns:
        filename (str): single tensor file  saved on the disk. Same location than input file.
    """

    import os

    import nibabel as nib
    import torch

    image_array = nib.load(input_img).get_fdata(dtype="float32")
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
    # make sure the tensor type is torch.float32
    output_file = (
        os.path.basename(input_img).replace(".nii.gz", ".pt"),
        image_tensor.clone(),
    )

    return [output_file]


############
# ROI    #
############
def check_mask_list(masks_location, roi_list, mask_pattern, cropping):
    import nibabel as nib
    import numpy as np

    for roi in roi_list:
        roi_path, desc = find_mask_path(masks_location, roi, mask_pattern, cropping)
        if roi_path is None:
            raise ValueError(
                f"The ROI '{roi}' does not correspond to a mask in the CAPS directory. {desc}"
            )
        roi_mask = nib.load(roi_path).get_fdata()
        mask_values = set(np.unique(roi_mask))
        if mask_values != {0, 1}:
            raise ValueError(
                "The ROI masks used should be binary (composed of 0 and 1 only)."
            )


def find_mask_path(
    masks_location: str, roi: str, mask_pattern: str, cropping: bool
) -> Tuple[str, str]:
    """
    Finds masks corresponding to the pattern asked and containing the adequate cropping description

    Args:
        masks_location: directory containing the masks.
        roi: name of the region.
        mask_pattern: pattern which should be found in the filename of the mask.
        cropping: if True the original image should contain the substring 'desc-Crop'.

    Returns:
        path of the mask or None if nothing was found.
        a human-friendly description of the pattern looked for.
    """
    from glob import glob
    from os import path

    # Check that pattern begins and ends with _ to avoid mixing keys
    if mask_pattern is None:
        mask_pattern = ""
    elif len(mask_pattern) != 0:
        if not mask_pattern.endswith("_"):
            mask_pattern += "_"
        if not mask_pattern[0] == "_":
            mask_pattern = "_" + mask_pattern

    candidates_pattern = path.join(
        masks_location, f"*{mask_pattern}*_roi-{roi}_mask.nii*"
    )
    desc = f"The mask should follow the pattern {candidates_pattern}. "
    candidates = glob(candidates_pattern)
    if cropping is None:
        pass
    elif cropping:
        candidates = [mask for mask in candidates if "_desc-Crop_" in mask]
        desc += f"and contain '_desc-Crop_' string."
    else:
        candidates = [mask for mask in candidates if "_desc-Crop_" not in mask]
        desc += f"and not contain '_desc-Crop_' string."

    if len(candidates) == 0:
        return None, desc
    else:
        return min(candidates, key=len), desc


def compute_output_pattern(mask_path, crop_output):
    """
    Computes the output pattern of the region cropped (without the source file prefix)
    Args:
        mask_path: path to the masks
        crop_output: If True the output is cropped, and the descriptor CropRoi must exist
    Returns:
        the output pattern
    """
    from os import path

    mask_filename = path.basename(mask_path)
    template_id = mask_filename.split("_")[0].split("-")[1]
    mask_descriptors = mask_filename.split("_")[1:-2:]
    roi_id = mask_filename.split("_")[-2].split("-")[1]
    if "desc-Crop" not in mask_descriptors and crop_output:
        mask_descriptors = ["desc-CropRoi"] + mask_descriptors
    elif "desc-Crop" in mask_descriptors:
        mask_descriptors = [
            descriptor for descriptor in mask_descriptors if descriptor != "desc-Crop"
        ]
        if crop_output:
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
    nii_path: str,
    masks_location: str,
    mask_pattern: str,
    cropped_input: bool,
    roi_names: List[str],
    uncrop_output: bool,
) -> List[Tuple[str, torch.Tensor]]:
    """Extracts regions of interest defined by masks
    This function extracts regions of interest from preprocessed nifti images.
    The regions are defined using binary masks that must be located in the CAPS
    at `masks/tpl-<template>`.
    Args:
        nii_path: path to the NifTi input image.
        masks_location: path to the masks
        mask_pattern: pattern to identify the masks
        cropped_input: if the input is cropped or not (contains desc-Crop)
        roi_names: list of the names of the regions that will be extracted.
        uncrop_output: if True, the final region is not cropped.
    Returns:
        list of tuples containing the path to the extracted ROI
            and the tensor of the corresponding ROI.
    """
    import nibabel as nib

    image_array = nib.load(nii_path).get_fdata(dtype="float32")
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

    roi_list = []
    for roi_name in roi_names:
        # read mask
        mask_path, _ = find_mask_path(
            masks_location, roi_name, mask_pattern, cropped_input
        )
        mask_np = nib.load(mask_path).get_fdata()

        roi_tensor = extract_roi_tensor(image_tensor, mask_np, uncrop_output)
        roi_path = extract_roi_path(nii_path, mask_np, uncrop_output)

        roi_list.append((roi_path, roi_tensor))

    return roi_list


def extract_roi_tensor(
    image_tensor: torch.Tensor,
    mask_np,
    uncrop_output: bool,
) -> torch.Tensor:

    if len(mask_np.shape) == 3:
        mask_np = np.expand_dims(mask_np, axis=0)
    elif len(mask_np.shape) == 4:
        assert mask_np[0] == 1
    else:
        raise ValueError(
            "ROI masks must be 3D or 4D tensors. "
            f"The dimension of your ROI mask is {len(mask_np.shape)}."
        )

    roi_tensor = image_tensor * mask_np
    if not uncrop_output:
        roi_tensor = roi_tensor[
            np.ix_(
                mask_np.any((1, 2, 3)),
                mask_np.any((0, 2, 3)),
                mask_np.any((0, 1, 3)),
                mask_np.any((0, 1, 2)),
            )
        ]
    return roi_tensor.float().clone()


def extract_roi_path(img_path: str, mask_path: str, uncrop_output: bool) -> str:
    input_img_filename = path.basename(img_path)

    sub_ses_prefix = "_".join(input_img_filename.split("_")[0:3:])
    if not sub_ses_prefix.endswith("_T1w"):
        sub_ses_prefix = "_".join(input_img_filename.split("_")[0:2:])
    input_suffix = input_img_filename.split("_")[-1].split(".")[0]

    output_pattern = compute_output_pattern(mask_path, not uncrop_output)

    return f"{sub_ses_prefix}_{output_pattern}_{input_suffix}.pt"


TEMPLATE_DICT = {
    "t1-linear": "MNI152NLin2009cSym",
    "pet-linear": "MNI152NLin2009cSym",
}

PATTERN_DICT = {
    "t1-linear": "res-1x1x1",
    "pet-linear": "res-1x1x1",
}
