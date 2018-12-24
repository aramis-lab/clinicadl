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

def get_subid_sesid_datasink(participant_id, session_id, caps_directory):
    """
    This is to extract the base_directory for the DataSink including participant_id and sesion_id in CAPS directory, also the tuple_list for substitution
    :param participant_id:
    :return: base_directory for DataSink
    """
    import os

    ## for MapNode
    base_directory = os.path.join(caps_directory, 'subjects', participant_id, session_id, 't1',
                                          'preprocessing_dl')

    subst_tuple_list = [  # registration
        (participant_id + '_' + session_id + '_SyN_QuickWarped_cropped_intensity_norm.nii.gz',
         participant_id + '_' + session_id + '_space-MNI_res-1x1x1_intensity_norm.nii.gz'),
        (participant_id + '_' + session_id + '_SyN_QuickWarped_cropped.nii.gz',
        participant_id + '_' + session_id + '_space-MNI_res-1x1x1.nii.gz'),
        (participant_id + '_' + session_id + '_SyN_QuickWarped_cropped.pt',
         participant_id + '_' + session_id + '_space-MNI_res-1x1x1.nii.pt'),
        (participant_id + '_' + session_id + '_SyN_QuickWarped.nii.gz',
         participant_id + '_' + session_id + '_space-MNI_res-1x1x1_linear_registration.nii.gz')
        ]

    regexp_substitutions = [
        # I don't know why it's adding this empty folder, so I remove it:
        # NOTE, . means any characters and * means any number of repetition in python regex
        (r'/out_file_crop/_cropnifti\d{1,4}/', r'/'),
        (r'/out_file_inn/_intensitynormalization\d{1,4}/', r'/'),
        (r'/out_file_reg/_antsRegistrationSyNQuick\d{1,4}/', r'/'),
        (r'/out_pt/_cropnifti\d{1,4}/', r'/'),
        # I don't know why it's adding this empty folder, so I remove it:
        (r'trait_added/_datasinker\d{1,4}/', r'')
    ]

    return base_directory, subst_tuple_list, regexp_substitutions

def bids_datagrabber_t1w(input_dir, subject_list, session_list):
    """
        Fetch t1 images from a BIDS directory based on subject_list and a session_list
    Args:
        input_dir: BIDS directory
        subject_list: a list containing all the participant_id
        session_list: a list containing all the session_id

    Returns: a list containing all the t1 images

    """
    from bids.grabbids.bids_layout import BIDSLayout
    from clinica.utils.stream import cprint
    import os

    bidslayout = BIDSLayout(input_dir)
    missing_subject_session = []
    anat_t1 = []
    if not bidslayout.get(target='run', return_type='id', type='T1w'):
        cprint("There is just one run for T1w image of this analysis")
        for i in range(len(subject_list)):
            t1 = bidslayout.get(return_type='file',
                                            type='T1w',
                                            extensions=['nii|nii.gz'],
                                            session=session_list[i].replace('ses-', ''),
                                            subject=subject_list[i].replace('sub-', ''))
            if len(t1) == 0:
                missing_subject_session.append([subject_list[i], session_list[i]])
            else:
                anat_t1.append(t1)
    else:
        cprint("There are more than one runs for T1w image for this analysis")
        for i in range(len(subject_list)):
            t1 = bidslayout.get(return_type='file',
                                            type='T1w',
                                            extensions=['nii|nii.gz'],
                                            session=session_list[i].replace('ses-', ''),
                                            subject=subject_list[i].replace('sub-', ''),
                                            run='1')
            if len(t1) == 0:
                missing_subject_session.append([subject_list[i], session_list[i]])
            else:
                anat_t1.append(t1)

    ### check if pybids works well tp find all the T1 images
    if len(missing_subject_session) > 0:
        error_string = 'Please verify there is no error in your tsv file. Clinica could not find T1 for those ' + str(len(missing_subject_session)) + ' subjects - session :'
        for e in missing_subject_session:
            error_string += '\n' + e[0] + ' with session ' + e[1]
        raise IOError(error_string)
    if len(anat_t1) != len(subject_list) or len(anat_t1) != len(session_list):
        raise ValueError('Pybids found ' + str(len(anat_t1)) + '  T1 but there are ' + str(len(subject_list)) + ' subjects !!! ')

    anat_t1 = [i[0] for i in anat_t1]

    ### check if each anat exist
    for anat in anat_t1:
        if os.path.isfile(anat):
            cprint("GREAT")
        else:
            raise Exception('This anat does not exist: %s' % anat)

    return anat_t1

def crop_nifti(input_img, ref_img):
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

def rename_file(subject, session):
    """
    Rename the output file for prefix
    :param subject:
    :param session:
    :return:
    """
    output_file_name = subject + '_' + session + '.nii.gz'

    return output_file_name

def ants_registration_syn_quick(fix_image, moving_image, participant_id, session_id):
    """
    This is a function to do rigid + affine registration using ants
    :param fix_image:
    :param moving_image:
    :param participant_id:
    :param session_id:
    :return:
    """

    import os
    import os.path as op

    image_warped = op.abspath(participant_id + '_' + session_id + '_SyN_QuickWarped.nii.gz')
    affine_matrix = op.abspath(participant_id + '_' + session_id + '_SyN_Quick0GenericAffine.mat')
    warp = op.abspath(participant_id + '_' + session_id + '_SyN_Quick1Warp.nii.gz')
    inverse_warped = op.abspath(participant_id + '_' + session_id + '_SyN_QuickInverseWarped.nii.gz')
    inverse_warp = op.abspath(participant_id + '_' + session_id + '_SyN_Quick1InverseWarp.nii.gz')

    cmd = 'antsRegistrationSyNQuick.sh -t a -d 3 -f ' + fix_image + ' -m ' + moving_image + ' -o ' + participant_id + '_' + session_id + '_SyN_Quick'
    os.system(cmd)

    return image_warped, affine_matrix, warp, inverse_warped, inverse_warp

def ants_histogram_intensity_normalization(crop_template, input_img, image_dimension):
    """
        This is a function to do histogram-based intensity normalization
    :param crop_template:
    :param input_img:
    :param image_dimension:
    :return:
    """

    import os

    basedir = os.getcwd()
    output_img = os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_intensity_norm.nii.gz')


    cmd = 'ImageMath ' + str(image_dimension) + ' ' + output_img + ' HistogramMatch ' + input_img + ' ' + crop_template
    os.system(cmd)

    return output_img

def get_caps_list(caps_directory, tsv):
    """
    THis is a function to grab all the cropped files
    :param caps_directory:
    :param tsv:
    :return:
    """
    import pandas as pd
    import os

    caps_intensity_nor_list = []
    df = pd.read_csv(tsv, sep='\t')
    if ('diagnosis' != list(df.columns.values)[2]) and ('session_id' != list(df.columns.values)[1]) and (
                'participant_id' != list(df.columns.values)[0]):
        raise Exception('the data file is not in the correct format.')
    img_list = list(df['participant_id'])
    sess_list = list(df['session_id'])

    for i in range(len(img_list)):
        img_path = os.path.join(caps_directory, 'subjects', img_list[i], sess_list[i], 't1', 'preprocessing_dl', img_list[i] + '_' + sess_list[i] + '_space-MNI_res-1x1x1_linear_registration.nii.gz')
        caps_intensity_nor_list.append(img_path)

    return caps_intensity_nor_list


def rank_mean(similarity, tsv, caps_directory):
    """
     This is a function to rank the difference img for all individuals
    :param similarity: a list of list containing the similarity value of each MRI
    :param tsv:
    :param caps_directory:
    :return:
    """

    import pandas as pd
    import os

    df = pd.read_csv(tsv, sep='\t')
    if ('diagnosis' != list(df.columns.values)[2]) and ('session_id' != list(df.columns.values)[1]) and (
                'participant_id' != list(df.columns.values)[0]):
        raise Exception('the data file is not in the correct format.')

    similarity = [abs(i[0]) for i in similarity]

    similaritys = pd.Series(similarity)
    df_new = df.copy()
    df_new['similarity'] = similaritys.values
    df_new[['similarity']] = df_new[['similarity']].astype(float)
    # df_new.sort_values('similarity')
    df_new.to_csv(os.path.join(caps_directory, 'qc_similarity.tsv'), sep='\t')


def save_as_pt(input_img):
    """
    This function is to transfer nii.gz file into .pt format, in order to train the pytorch model more efficient when loading the data.
    :param input_img:
    :return:
    """

    import torch, os
    import nibabel as nib

    image_array = nib.load(input_img).get_fdata()
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
    output_file = os.path.join(os.path.dirname(input_img), input_img.split('.nii.gz')[0] + '.pt')
    # save
    torch.save(image_tensor, output_file)

    return output_file





