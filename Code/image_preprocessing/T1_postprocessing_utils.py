# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:17:41 2017

@author: Junhao WEN
"""

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def get_caps_t1(caps_directory, tsv):
    """
    THis is a function to grab all the cropped files
    :param caps_directory:
    :param tsv:
    :return:
    """
    import pandas as pd
    import os

    preprocessed_T1 = []
    df = pd.read_csv(tsv, sep='\t')
    if ('session_id' != list(df.columns.values)[1]) and (
                'participant_id' != list(df.columns.values)[0]):
        raise Exception('the data file is not in the correct format.')
    img_list = list(df['participant_id'])
    sess_list = list(df['session_id'])

    for i in range(len(img_list)):
        img_path = os.path.join(caps_directory, 'subjects', img_list[i], sess_list[i], 't1', 'preprocessing_dl', img_list[i] + '_' + sess_list[i] + '_space-MNI_res-1x1x1.nii.gz')
        preprocessed_T1.append(img_path)

    return preprocessed_T1

def extract_slices(preprocessed_T1, slice_direction=0, slice_mode='original'):
    """
    This is to extract the slices from three directions
    :param preprocessed_T1:
    :param slice_direction: which axis direction that the slices were extracted
    :return:
    """
    import torch, os

    image_tensor = torch.load(preprocessed_T1)
    ## reshape the tensor, delete the first dimension for slice-level
    image_tensor = image_tensor.view(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3])

    ## sagital
    slice_list_sag = range(20, image_tensor.shape[0] - 20) # delete the first 20 slice and last 20 slices

    if slice_direction == 0:
        for index_slice in slice_list_sag:
            # for i in slice_list:
            ## sagital
            slice_select_sag = image_tensor[index_slice, :, :]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_sag = slice_select_sag.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_sag = (slice_select_sag - slice_select_sag.min()) / (slice_select_sag.max() - slice_select_sag.min())
            extracted_slice_rgb_sag = torch.stack((slice_select_sag, slice_select_sag, slice_select_sag)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-sag_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_sag, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-sag_rgbslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_sag, output_file_rgb)

    elif slice_direction == 1:
        ## cornal
        slice_list_cor = range(15, image_tensor.shape[1] - 15) # delete the first 20 slice and last 15 slices
        for index_slice in slice_list_cor:
            # for i in slice_list:
            ## sagital
            slice_select_cor = image_tensor[:, index_slice, :]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_cor = slice_select_cor.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_cor = (slice_select_cor - slice_select_cor.min()) / (slice_select_cor.max() - slice_select_cor.min())
            extracted_slice_rgb_cor = torch.stack((slice_select_cor, slice_select_cor, slice_select_cor)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-cor_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_cor, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-cor_rgblslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_cor, output_file_rgb)

    else:

        ## axial
        slice_list_axi = range(15, image_tensor.shape[2] - 15) # delete the first 20 slice and last 15 slices
        for index_slice in slice_list_axi:
            # for i in slice_list:
            ## sagital
            slice_select_axi = image_tensor[:, :, index_slice]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_axi = slice_select_axi.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_axi = (slice_select_axi - slice_select_axi.min()) / (slice_select_axi.max() - slice_select_axi.min())
            extracted_slice_rgb_axi = torch.stack((slice_select_axi, slice_select_axi, slice_select_axi)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-axi_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_axi, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-axi_rgblslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_axi, output_file_rgb)

    return preprocessed_T1

def extract_patches(preprocessed_T1, patch_size, stride_size):
    """
    This is to extract the patches from three directions
    :param preprocessed_T1:
    :return:
    """
    import torch, os

    image_tensor = torch.load(preprocessed_T1)

    ## use pytorch tensor.upfold to crop the patch.
    patches_tensor = image_tensor.unfold(1, patch_size, stride_size).unfold(2, patch_size, stride_size).unfold(3, patch_size, stride_size).contiguous()
    # the dimension of patch_tensor should be [1, patch_num1, patch_num2, patch_num3, patch_size1, patch_size2, patch_size3]
    patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)

    for index_patch in xrange(patches_tensor.shape[0]):
        extracted_patch = patches_tensor[index_patch, ...].unsqueeze_(0) ## add one dimension
        # save into .pt format
        output_patch = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_patchsize-' + str(patch_size) + '_stride-' + str(stride_size) + '_patch-' + str(index_patch) + '.pt')
        torch.save(extracted_patch, output_patch)

    return preprocessed_T1

def save_as_pt(input_img):
    """
    This function is to transfer nii.gz file into .pt format, in order to train the pytorch model more efficient when loading the data.
    :param input_img:
    :return:
    """

    import torch, os
    import nibabel as nib

    image_array = nib.load(input_img).get_fdata()
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
    ## make sure the tensor dtype is torch.float32
    output_file = os.path.join(os.path.dirname(input_img), input_img.split('.nii.gz')[0] + '.pt')
    # save
    torch.save(image_tensor, output_file)

    return output_file