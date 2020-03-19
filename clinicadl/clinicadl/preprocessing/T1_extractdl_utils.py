
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

    import os
    import re
    m = re.search(r'(sub-[a-zA-Z0-9]+)/(ses-[a-zA-Z0-9]+)', bids_or_caps_filename)
    if m is None:
        raise ValueError('Input filename is not in a BIDS or CAPS compliant format.'
                'It does not contain the participant and session ID.')
    subject = m.group(1)
    session = m.group(2)
    return os.path.join('subjects', subject, session)

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
