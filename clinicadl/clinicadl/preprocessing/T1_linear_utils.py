# coding: utf8

# Get containers to ptoduce the CAPS structure


def container_from_filename(bids_or_caps_filename):
    """Extract container from BIDS or CAPS file.
    Args:
       bids_or_caps_filename (str): full path to BIDS or CAPS filename.
    Returns:
       Container path of the form "subjects/<participant_id>/<session_id>"
    Examples:
       >>> from clinica.utils.nipype import container_from_filename
       >>> container_from_filename('/path/to/bids/sub-CLNC01/ses-M00/anat/sub-CLNC01_ses-M00_T1w.nii.gz')
               'subjects/sub-CLNC01/ses-M00'
       >>> container_from_filename('caps/subjects/sub-CLNC01/ses-M00/dwi/preprocessing/sub-CLNC01_ses-M00_preproc.nii')
               'subjects/sub-CLNC01/ses-M00'
    """

    from os.path import join
    import re
    m = re.search(r'(sub-[a-zA-Z0-9]+)/(ses-[a-zA-Z0-9]+)', bids_or_caps_filename)
    if m is None:
        raise ValueError('Input filename is not in a BIDS or CAPS compliant format.'
                         'It does not contain the participant and session ID.')
    subject = m.group(1)
    session = m.group(2)
    return join('subjects', subject, session)


def get_data_datasink(image_id):
    substitutions_ls = [  # registration
            (image_id + '_T1w_corrected.nii.gz',
                image_id + '_T1w_desc-BiasCorrected_T1w.nii.gz'),
            (image_id + 'Warped_cropped_intensity_norm.nii.gz',
                image_id + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_intensity_norm_T1w.nii.gz'),
            (image_id + 'Warped_cropped.nii.gz',
                image_id + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'),
            (image_id + '0GenericAffine.mat',
                image_id + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_affine.mat'),
            (image_id + 'Warped_cropped.pt',
                image_id + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt'),
            (image_id + 'Warped.nii.gz',
                image_id + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz')
            ]
    return image_id, substitutions_ls


def crop_nifti(input_img, ref_crop):
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
    from nilearn.image import resample_img, resample_to_img
    from nibabel.spatialimages import SpatialImage

    basedir = os.getcwd()
    # crop_ref = crop_img(ref_img, rtol=0.5)
    # crop_ref.to_filename(os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped_template.nii.gz'))
    # crop_template = os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped_template.nii.gz')

    # resample the individual MRI onto the cropped template image
    crop_img = resample_to_img(input_img, ref_crop, force_resample=True)
    crop_img.to_filename(os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped.nii.gz'))

    output_img = os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped.nii.gz')
    crop_template = ref_crop

    return output_img, crop_template


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
