"""
Produces a tsv file to study all the nii files and perform the quality check.
"""
from os import path
import numpy as np
import pandas as pd
import nibabel as nib
import os
from pathlib import Path

from clinicadl.tools.inputs.input import fetch_file, RemoteFileStructure


def extract_metrics(caps_dir, output_dir):
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # Load eyes segmentation
    home = str(Path.home())
    cache_clinicadl = path.join(home, '.cache', 'clinicadl', 'segmentation')
    url_aramis = 'https://aramislab.paris.inria.fr/files/data/template/'
    FILE1 = RemoteFileStructure(
        filename='eyes_segmentation.nii.gz',
        url=url_aramis,
        checksum='56f699c06cafc62ad8bb5b41b188c7c412d684d810a11d6f4cbb441c0ce944ee'
    )

    if not(path.exists(cache_clinicadl)):
        os.makedirs(cache_clinicadl)

    segmentation_file = path.join(cache_clinicadl, FILE1.filename)

    if not(path.exists(segmentation_file)):
        try:
            segmentation_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            raise IOError(
                'Unable to download required eyes segmentation for QC:', err)

    segmentation_nii = nib.load(segmentation_file)
    segmentation_np = segmentation_nii.get_fdata()

    # Get the GM template
    template_path = path.join(
        caps_dir, 'groups', 'group-ADNIbl', 't1', 'group-ADNIbl_template.nii.gz')
    template_nii = nib.load(template_path)
    template_np = template_nii.get_fdata()
    template_np = np.sum(template_np, axis=3)
    template_segmentation_np = template_np * segmentation_np

    # Get the data
    filename = path.join(output_dir, 'QC_metrics.tsv')
    columns = ['participant_id', 'session_id', 'max_intensity',
               'non_zero_percentage', 'frontal_similarity']
    results_df = pd.DataFrame()

    subjects = os.listdir(path.join(caps_dir, 'subjects'))
    subjects = [subject for subject in subjects if subject[:4:] == "sub-"]
    for subject in subjects:
        subject_path = path.join(caps_dir, 'subjects', subject)
        sessions = os.listdir(subject_path)
        sessions = [session for session in sessions if session[:4:] == "ses-"]
        for session in sessions:
            image_path = path.join(subject_path, session, 't1', 'spm', 'segmentation', 'normalized_space',
                                   subject + '_' + session + '_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.nii.gz')

            if path.exists(image_path):
                # GM analysis
                image_nii = nib.load(image_path)
                image_np = image_nii.get_fdata()
                image_segmentation_np = image_np * segmentation_np
                eyes_nmi_value = nmi(
                    occlusion1=template_segmentation_np, occlusion2=image_segmentation_np)

                non_zero_percentage = np.count_nonzero(
                    image_np) / image_np.size

                row = [[subject, session, np.max(
                    image_np), non_zero_percentage, eyes_nmi_value]]
                row_df = pd.DataFrame(row, columns=columns)
                results_df = pd.concat([results_df, row_df])

    results_df.sort_values('max_intensity', inplace=True, ascending=True)
    results_df.to_csv(filename, sep='\t', index=False)


def nmi(occlusion1, occlusion2):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    hist_inter, _, _ = np.histogram2d(occlusion1.ravel(), occlusion2.ravel())
    hist1, _, _ = np.histogram2d(occlusion1.ravel(), occlusion1.ravel())
    hist2, _, _ = np.histogram2d(occlusion2.ravel(), occlusion2.ravel())

    return 2 * _mutual_information(hist_inter) / (
        _mutual_information(hist1) + _mutual_information(hist2))


def _mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
