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

def crop_nifti_hippo(input_img, ref_img):
    """

    :param input_img:
    :param crop_sagittal:
    :param crop_coronal:
    :param crop_axial:
    :return:
    """

    import nibabel as nib
    import os
    import numpy as np
    from nilearn.image import resample_img, crop_img, resample_to_img
    from nibabel.spatialimages import SpatialImage

    basedir = os.getcwd()
    crop_ref = crop_img(ref_img, rtol=0.5)
    crop_ref.to_filename(os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped_template.nii.gz'))
    crop_template = os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped_template.nii.gz')

    ## resample the individual MRI onto the cropped template image
    crop_img = resample_to_img(input_img, crop_template)
    crop_img.to_filename(os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped.nii.gz'))

    output_img = os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped.nii.gz')

    return output_img, crop_template
