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
    cropped_hipp_file_name = []
    preprocessed_T1_folder = []

    df = pd.read_csv(tsv, sep='\t')
    if ('session_id' != list(df.columns.values)[1]) and (
                'participant_id' != list(df.columns.values)[0]):
        raise Exception('the data file is not in the correct format.')
    participant_id = list(df['participant_id'])
    session_id = list(df['session_id'])

    for i in range(len(participant_id)):
        img_path = os.path.join(caps_directory, 'subjects', participant_id[i], session_id[i], 't1', 'preprocessing_dl', participant_id[i] + '_' + session_id[i] + '_space-MNI_res-1x1x1.nii.gz')
        preprocessed_T1.append(img_path)
        preprocessed_T1_folder.append(os.path.join(caps_directory, 'subjects', participant_id[i], session_id[i], 't1', 'preprocessing_dl'))
        cropped_hipp_file_name.append(participant_id[i] + '_' + session_id[i] + '_space-MNI_res-1x1x1_hippocampus.nii')

    return preprocessed_T1, cropped_hipp_file_name, participant_id, session_id, preprocessed_T1_folder


def save_as_pt(input_img):
    """
    This function is to transfer nii.gz file into .pt format, in order to train the classifiers model more efficient when loading the data.
    :param input_img:
    :return:
    """

    import torch
    import os
    import nibabel as nib

    image_array = nib.load(input_img).get_fdata()
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
    # make sure the tensor dtype is torch.float32
    output_file = os.path.join(os.path.dirname(input_img), input_img.split('.nii')[0] + '.pt')
    # save
    torch.save(image_tensor.clone(), output_file)

    return output_file


def compress_nii(in_file, same_dir=True):
    """
        This is a function to compress the resulting nii images
    Args:
        in_file:

    Returns:

    """
    from os import getcwd, remove
    from os.path import abspath, join
    import gzip
    import shutil
    from nipype.utils.filemanip import split_filename

    orig_dir, base, ext = split_filename(str(in_file))

    # Not compressed
    if same_dir:
        out_file = abspath(join(orig_dir, base + ext + '.gz'))
    else:
        out_file = abspath(join(getcwd(), base + ext + '.gz'))

    with open(in_file, 'rb') as f_in, gzip.open(out_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_file


def get_subid_sesid_datasink(participant_id, session_id, caps_directory, hemi):
    """
    This is to extract the base_directory for the DataSink including participant_id and sesion_id in CAPS directory, also the tuple_list for substitution
    :param participant_id:
    :return: base_directory for DataSink
    """
    import os

    # for MapNode
    base_directory = os.path.join(
            caps_directory, 'subjects', participant_id,
            session_id, 't1', 'preprocessing_dl')

    subst_tuple_list = [
        (participant_id + '_' + session_id + '_space-MNI_res-1x1x1_hippocampus.nii.gz',
         participant_id + '_' + session_id + '_space-MNI_res-1x1x1_hippocampus_hemi-' + hemi + '.nii.gz'),
        (participant_id + '_' + session_id + '_space-MNI_res-1x1x1_hippocampus.pt',
         participant_id + '_' + session_id + '_space-MNI_res-1x1x1_hippocampus_hemi-' + hemi + '.pt'),
        ]

    regexp_substitutions = [
        # I don't know why it's adding this empty folder, so I remove it:
        # NOTE, . means any characters and * means any number of repetition in python regex
        (r'/output_hippocampus_nii/_hippocampus_patches\d{1,4}/', r'/'),
        (r'/output_hippocampus_pt/_hippocampus_patches\d{1,4}/', r'/'),
        # I don't know why it's adding this empty folder, so I remove it:
        (r'/trait_added/_datasinker\d{1,4}/', r'/')
    ]

    return base_directory, subst_tuple_list, regexp_substitutions
