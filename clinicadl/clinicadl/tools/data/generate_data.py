"""
This file generates data for trivial or intractable (random) data for binary classification.
"""
import pandas as pd
import numpy as np
import nibabel as nib
from os import path
import os
import torch.nn.functional as F
import torch
from .utils import im_loss_roi_gaussian_distribution, find_image_path
from ..tsv.tsv_utils import baseline_df


def generate_random_dataset(caps_dir, tsv_path, output_dir, n_subjects, mean=0, sigma=0.5,
                            preprocessing="linear", output_size=None):
    """
    Generates a random dataset for intractable classification task from the first subject of the tsv file.

    :param caps_dir: (str) path to the CAPS directory.
    :param tsv_path: (str) path to tsv file of list of subjects/sessions.
    :param output_dir: (str) folder containing the synthetic dataset in CAPS format
    :param n_subjects: (int) number of subjects in each class of the synthetic dataset
    :param mean: (float) mean of the gaussian noise
    :param sigma: (float) standard deviation of the gaussian noise
    :param preprocessing: (str) preprocessing performed. Must be in ['linear', 'extensive'].
    :param output_size: (tuple[int]) size of the output. If None no interpolation will be performed.
    """
    # Read DataFrame
    data_df = pd.read_csv(tsv_path, sep='\t')

    # Create subjects dir
    if not path.exists(path.join(output_dir, 'subjects')):
        os.makedirs(path.join(output_dir, 'subjects'))

    # Retrieve image of first subject
    participant_id = data_df.loc[0, 'participant_id']
    session_id = data_df.loc[0, 'session_id']

    image_path = find_image_path(caps_dir, participant_id, session_id, preprocessing)
    image_nii = nib.load(image_path)
    image = image_nii.get_data()

    # Create output tsv file
    participant_id_list = ['sub-RAND%i' % i for i in range(2 * n_subjects)]
    session_id_list = ['ses-M00'] * 2 * n_subjects
    diagnosis_list = ['AD'] * n_subjects + ['CN'] * n_subjects
    data = np.array([participant_id_list, session_id_list, diagnosis_list])
    data = data.T
    output_df = pd.DataFrame(data, columns=['participant_id', 'session_id', 'diagnosis'])
    output_df['age'] = 60
    output_df['sex'] = 'F'
    output_df.to_csv(path.join(output_dir, 'data.tsv'), sep='\t', index=False)

    for i in range(2 * n_subjects):
        gauss = np.random.normal(mean, sigma, image.shape)
        participant_id = 'sub-RAND%i' % i
        noisy_image = image + gauss
        if output_size is not None:
            noisy_image_pt = torch.Tensor(noisy_image[np.newaxis, np.newaxis, :])
            noisy_image_pt = F.interpolate(noisy_image_pt, output_size)
            noisy_image = noisy_image_pt.numpy()[0, 0, :, :, :]
        noisy_image_nii = nib.Nifti1Image(noisy_image, header=image_nii.header, affine=image_nii.affine)
        noisy_image_nii_path = path.join(output_dir, 'subjects', participant_id, 'ses-M00', 't1', 'preprocessing_dl')
        noisy_image_nii_filename = participant_id + '_ses-M00_space-MNI_res-1x1x1.nii.gz'
        if not path.exists(noisy_image_nii_path):
            os.makedirs(noisy_image_nii_path)
        nib.save(noisy_image_nii, path.join(noisy_image_nii_path, noisy_image_nii_filename))


def generate_trivial_dataset(caps_dir, tsv_path, output_dir, n_subjects, preprocessing="linear",
                             mask_path=None, atrophy_percent=60, output_size=None, group=None):
    """
    Generates a fully separable dataset.

    :param caps_dir: (str) path to the CAPS directory.
    :param tsv_path: (str) path to tsv file of list of subjects/sessions.
    :param output_dir: (str) folder containing the synthetic dataset in CAPS format.
    :param n_subjects: (int) number of subjects in each class of the synthetic dataset
    :param preprocessing: (str) preprocessing performed. Must be in ['linear', 'extensive'].
    :param mask_path: (str) path to the extracted masks to generate the two labels
    :param atrophy_percent: (float) percentage of atrophy applied
    :param output_size: (tuple[int]) size of the output. If None no interpolation will be performed.
    :param group: (str) group used for dartel preprocessing.
    """

    # Read DataFrame
    data_df = pd.read_csv(tsv_path, sep='\t')
    data_df = baseline_df(data_df, "None")

    if n_subjects > len(data_df):
        raise ValueError("The number of subjects %i cannot be higher than the number of subjects in the baseline "
                         "DataFrame extracted from %s" % (n_subjects, tsv_path))

    if mask_path is None:
        raise ValueError('Please provide a path to masks. Such masks are available at '
                         'clinicadl/tools/data/AAL2.')

    # Output tsv file
    columns = ['participant_id', 'session_id', 'diagnosis', 'age', 'sex']
    output_df = pd.DataFrame(columns=columns)
    diagnosis_list = ["AD", "CN"]

    for i in range(2 * n_subjects):
        data_idx = i // 2
        label = i % 2

        participant_id = data_df.loc[data_idx, "participant_id"]
        session_id = data_df.loc[data_idx, "session_id"]
        filename = 'sub-TRIV%i_ses-M00_space-MNI_res-1x1x1.nii.gz' % i
        path_image = os.path.join(output_dir, 'subjects', 'sub-TRIV%i' % i, 'ses-M00', 't1', 'preprocessing_dl')

        if not os.path.exists(path_image):
            os.makedirs(path_image)

        image_path = find_image_path(caps_dir, participant_id, session_id, preprocessing, group)
        image_nii = nib.load(image_path)
        image = image_nii.get_data()

        atlas_to_mask = nib.load(os.path.join(mask_path, 'mask-%i.nii' % (label + 1))).get_data()

        # Create atrophied image
        trivial_image = im_loss_roi_gaussian_distribution(image, atlas_to_mask, atrophy_percent)
        if output_size is not None:
            trivial_image_pt = torch.Tensor(trivial_image[np.newaxis, np.newaxis, :])
            trivial_image_pt = F.interpolate(trivial_image_pt, output_size)
            trivial_image = trivial_image_pt.numpy()[0, 0, :, :, :]
        trivial_image_nii = nib.Nifti1Image(trivial_image, affine=image_nii.affine)
        trivial_image_nii.to_filename(os.path.join(path_image, filename))

        # Append row to output tsv
        row = ['sub-TRIV%i' % i, 'ses-M00', diagnosis_list[label], 60, 'F']
        row_df = pd.DataFrame([row], columns=columns)
        output_df = output_df.append(row_df)

    output_df.to_csv(path.join(output_dir, 'data.tsv'), sep='\t', index=False)
