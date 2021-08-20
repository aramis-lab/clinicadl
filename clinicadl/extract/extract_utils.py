# coding: utf8
from typing import Union


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
        "use_uncropped_image": save_features,
    }
    # use extracted features

    parameters["prepare_dl"] = save_features

    if modality == "custom":
        parameters["custom_suffix"] = custom_suffix
    if modality == "pet-linear":
        parameters["acq_label"] = acq_label
        parameters["suvr_reference_region"] = suvr_reference_region

    return parameters


def extract_slices(
    nii_path: str,
    slice_direction: int = 0,
    slice_mode: str = "single",
    discarded_slices: Union[int, tuple] = 0,
):
    """Extracts the slices from three directions
    This function extracts slices form the preprocessed nifti image.  The
    direction of extraction can be defined either on sagital direction (0),
    coronal direction (1) or axial direction (other). The output slices can be
    stores following two modes: single (1 channel) ou RGB (3 channels, all the
    same).
    Args:
        nii_path: nifti format MRI image.
        slice_direction: which axis direction that the slices were extracted
        slice_mode: 'single' or 'RGB'.
        discarded_slices: Number of slices to discard at the beginning and the end of the image.
            Will be a tuple of two integers if the number of slices to discard at the beginning
            and at the end differ.
    Returns:
        file: multiple tensors saved on the disk, suffixes corresponds to
            indexes of the slices. Same location than input file.
    """
    import os

    import nibabel as nib
    import torch

    direction_dict = {0: "sag", 1: "cor", 2: "axi"}

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
    slice_list = range(begin_discard, image_tensor.shape[slice_direction] - end_discard)

    input_img_filename = os.path.basename(nii_path)

    txt_idx = input_img_filename.rfind("_")
    it_filename_prefix = input_img_filename[0:txt_idx]
    it_filename_suffix = input_img_filename[txt_idx:]
    it_filename_suffix = it_filename_suffix.replace(".nii.gz", ".pt")

    output_slices = []
    for slice_index in slice_list:
        # Allow to select the slice `slice_index` in dimension `slice_direction`
        idx_tuple = tuple(
            [slice(None)] * slice_direction
            + [slice_index]
            + [slice(None)] * (2 - slice_direction)
        )
        slice_selected = image_tensor[idx_tuple]
        slice_selected.unsqueeze_(0)  # shape is 1 * W * L

        if slice_mode == "rgb":
            slice_selected = torch.cat(
                (slice_selected, slice_selected, slice_selected)
            )  # shape is 3 * W * L

        output_slices.append(
            (
                f"{it_filename_prefix}_axis-{direction_dict[slice_direction]}"
                f"_channel-{slice_mode}_slice-{slice_index}{it_filename_suffix}",
                slice_selected.clone(),
            )
        )

    return output_slices


def extract_patches(input_img, patch_size, stride_size):
    """Extracts the patches
    This function extracts patches form the preprocessed nifti image. Patch size
    if provided as input and also the stride size. If stride size is smaller
    than the patch size an overlap exist between consecutive patches. If stride
    size is equal to path size there is no overlap. Otherwise, unprocessed
    zones can exits.
    Args:
        input_img: nifti format MRI image.
        patch_size: size of a single patch.
        stride_size: size of the stride leading to next patch.
    Returns:
        file: multiple tensors saved on the disk, suffixes corresponds to
            indexes of the patches. Same location than input file.
    """
    import os

    import nibabel as nib
    import torch

    image_array = nib.load(input_img).get_fdata(dtype="float32")
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

    # use classifiers tensor.upfold to crop the patch.
    patches_tensor = (
        image_tensor.unfold(1, patch_size, stride_size)
        .unfold(2, patch_size, stride_size)
        .unfold(3, patch_size, stride_size)
        .contiguous()
    )
    # the dimension of patch_tensor should be [1, patch_num1, patch_num2, patch_num3, patch_size1, patch_size2, patch_size3]
    patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)

    input_img_filename = os.path.basename(input_img)
    txt_idx = input_img_filename.rfind("_")
    it_filename_prefix = input_img_filename[0:txt_idx]
    it_filename_suffix = input_img_filename[txt_idx:]
    it_filename_suffix = it_filename_suffix.replace(".nii.gz", ".pt")

    output_patch = []
    for index_patch in range(patches_tensor.shape[0]):
        extracted_patch = patches_tensor[index_patch, ...].unsqueeze_(
            0
        )  # add one dimension
        # save into .pt format
        output_patch.append(
            (
                it_filename_prefix
                + "_patchsize-"
                + str(patch_size)
                + "_stride-"
                + str(stride_size)
                + "_patch-"
                + str(index_patch)
                + it_filename_suffix,
                extracted_patch.clone(),
            )
        )
        # torch.save(extracted_patch.clone(), output_patch[index_patch])

    return output_patch


def extract_images(input_img):
    """Extract the images
    This function convert nifti image to tensor (.pt) version of the image.
    Tensor version is saved at the same location than input_img.
    Args:
        input_img: nifti format MRI image.
    Returns:
        filename (str): single tensor file  saved on the disk. Same location than input file.
    """

    import os

    import nibabel as nib
    import torch

    image_array = nib.load(input_img).get_fdata(dtype="float32")
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
    # make sure the tensor dtype is torch.float32
    output_file = (
        os.path.basename(input_img).split(".nii.gz")[0] + ".pt",
        image_tensor.clone(),
    )

    return [output_file]


# ROI extraction utils


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


def find_mask_path(masks_location, roi, mask_pattern, cropping):
    """Finds masks corresponding to the pattern asked and containing the adequate cropping description"""
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
    desc = f"The mask should follow the pattern {candidates_pattern} "
    candidates = glob(candidates_pattern)
    if cropping is None:
        desc += "."
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
    output_pattern = f"space-{template_id}_{mask_pattern}_roi-{roi_id}"

    return output_pattern


def extract_roi(
    input_img,
    masks_location,
    mask_pattern,
    cropped_input,
    roi_list,
    uncrop_output,
):
    """Extracts regions of interest defined by masks
    This function extracts regions of interest from preprocessed nifti images.
    The regions are defined using binary masks that must be located in the CAPS
    at `masks/tpl-<template>`.
    Args:
        input_img: nifti format MRI image.
        masks_location: path to the masks
        mask_pattern: pattern to identify the masks
        cropped_input: if the input is cropped or not (contains desc-Crop)
        roi_list: list of the names of the regions that will be extracted.
        uncrop_output: if True, the final region is not cropped.
    Returns:
        file: multiple tensors saved on the disk, suffixes corresponds to
            indexes of the patches. Same location than input file.
    """
    import os

    import nibabel as nib
    import numpy as np
    import torch

    image_array = nib.load(input_img).get_fdata(dtype="float32")
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

    input_img_filename = os.path.basename(input_img)

    sub_ses_prefix = "_".join(input_img_filename.split("_")[0:3:])
    if not sub_ses_prefix.endswith("_T1w"):
        sub_ses_prefix = "_".join(input_img_filename.split("_")[0:2:])
    input_suffix = input_img_filename.split("_")[-1].split(".")[0]

    output_roi = []
    for index_roi, roi in enumerate(roi_list):
        # read mask
        mask_path, _ = find_mask_path(masks_location, roi, mask_pattern, cropped_input)
        mask_np = nib.load(mask_path).get_fdata()
        if len(mask_np.shape) == 3:
            mask_np = mask_np[np.newaxis, :]

        extracted_roi = image_tensor * mask_np
        if not uncrop_output:
            extracted_roi = extracted_roi[
                np.ix_(
                    mask_np.any((1, 2, 3)),
                    mask_np.any((0, 2, 3)),
                    mask_np.any((0, 1, 3)),
                    mask_np.any((0, 1, 2)),
                )
            ]
        extracted_roi = extracted_roi.float()
        # save into .pt format
        output_pattern = compute_output_pattern(mask_path, not uncrop_output)
        output_roi.append(
            (
                f"{sub_ses_prefix}_{output_pattern}_{input_suffix}.pt",
                extracted_roi.clone(),
            )
        )

    return output_roi


TEMPLATE_DICT = {
    "t1-linear": "MNI152NLin2009cSym",
    "t1-extensive": "Ixi549Space",
    "pet-linear": "MNI152NLin2009cSym",
}
