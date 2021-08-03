# coding: utf8


def extract_slices(input_img, slice_direction=0, slice_mode="single"):
    """Extracts the slices from three directions
    This function extracts slices form the preprocessed nifti image.  The
    direction of extraction can be defined either on sagital direction (0),
    coronal direction (1) or axial direction (other). The output slices can be
    stores following two modes: single (1 channel) ou RGB (3 channels, all the
    same).
    Args:
        input_img: nifti format MRI image.
        slice_direction: which axis direction that the slices were extracted
        slice_mode: 'single' or 'RGB'.
    Returns:
        file: multiple tensors saved on the disk, suffixes corresponds to
            indexes of the slices. Same location than input file.
    """
    import os

    import nibabel as nib
    import torch

    image_array = nib.load(input_img).get_fdata(dtype="float32")
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
    # reshape the tensor, delete the first dimension for slice-level
    image_tensor = image_tensor.view(
        image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
    )

    # sagital
    # M and N correspond to the first and last slices (if need to remove)
    M = 0
    N = 0
    slice_list_sag = range(
        M, image_tensor.shape[0] - N
    )  # delete the first M slices and last N slices

    input_img_filename = os.path.basename(input_img)

    txt_idx = input_img_filename.rfind("_")
    it_filename_prefix = input_img_filename[0:txt_idx]
    it_filename_suffix = input_img_filename[txt_idx:]

    output_slices = []
    if slice_direction == 0:
        for index_slice, index_slice_list in zip(
            slice_list_sag, range(len(slice_list_sag))
        ):
            # for i in slice_list:
            # sagital
            slice_select_sag = image_tensor[index_slice, :, :]

            extracted_slice_original_sag = slice_select_sag.unsqueeze(
                0
            )  # shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_sag = (slice_select_sag - slice_select_sag.min()) / (
                slice_select_sag.max() - slice_select_sag.min()
            )
            extracted_slice_rgb_sag = torch.stack(
                (slice_select_sag, slice_select_sag, slice_select_sag)
            )  # shape should be 3 * W * L

            # save into .pt format
            if slice_mode == "single":
                output_slices.append(
                    (
                        it_filename_prefix
                        + "_axis-sag_channel-single_slice-"
                        + str(index_slice)
                        + it_filename_suffix,
                        extracted_slice_original_sag.clone(),
                    )
                )
            elif slice_mode == "rgb":
                output_slices.append(
                    (
                        it_filename_prefix
                        + "_axis-sag_channel-rgb_slice-"
                        + str(index_slice)
                        + it_filename_suffix,
                        extracted_slice_rgb_sag.clone(),
                    )
                )


    elif slice_direction == 1:
        # cornal
        slice_list_cor = range(
            M, image_tensor.shape[1] - N
        )  # delete the first M slices and last N slices
        for index_slice, index_slice_list in zip(
            slice_list_cor, range(len(slice_list_cor))
        ):
            # for i in slice_list:
            # sagital
            slice_select_cor = image_tensor[:, index_slice, :]

            extracted_slice_original_cor = slice_select_cor.unsqueeze(
                0
            )  # shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_cor = (slice_select_cor - slice_select_cor.min()) / (
                slice_select_cor.max() - slice_select_cor.min()
            )
            extracted_slice_rgb_cor = torch.stack(
                (slice_select_cor, slice_select_cor, slice_select_cor)
            )  # shape should be 3 * W * L

            # save into .pt format
            if slice_mode == "single":
                output_slices.append(
                    (
                        it_filename_prefix
                        + "_axis-cor_channel-single_slice-"
                        + str(index_slice)
                        + it_filename_suffix,
                        extracted_slice_original_cor.clone(),
                    )
                )
            elif slice_mode == "rgb":
                output_slices.append(
                    (
                        it_filename_prefix
                        + "_axis-cor_channel-rgb_slice-"
                        + str(index_slice)
                        + it_filename_suffix,
                        extracted_slice_rgb_cor.clone(),
                    )
                )

    else:

        # axial
        slice_list_axi = range(
            M, image_tensor.shape[2] - N
        )  # delete the first M slices and last N slices
        for index_slice, index_slice_list in zip(
            slice_list_axi, range(len(slice_list_axi))
        ):
            # for i in slice_list:
            # sagital
            slice_select_axi = image_tensor[:, :, index_slice]

            extracted_slice_original_axi = slice_select_axi.unsqueeze(
                0
            )  # shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_axi = (slice_select_axi - slice_select_axi.min()) / (
                slice_select_axi.max() - slice_select_axi.min()
            )
            extracted_slice_rgb_axi = torch.stack(
                (slice_select_axi, slice_select_axi, slice_select_axi)
            )  # shape should be 3 * W * L

            # save into .pt format
            if slice_mode == "single":
                output_slices.append(
                    (
                        it_filename_prefix
                        + "_axis-axi_channel-single_slice-"
                        + str(index_slice)
                        + it_filename_suffix,
                        extracted_slice_original_axi.clone(),
                    )
                )

            elif slice_mode == "rgb":
                output_slices.append(
                    (
                        it_filename_prefix
                        + "_axis-axi_channel-rgb_slice-"
                        + str(index_slice)
                        + it_filename_suffix,
                        extracted_slice_rgb_axi.clone(),
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
