# coding: utf8

"""
This file generates data for trivial or intractable (random) data for binary classification.
"""
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from os.path import join, exists
from os import makedirs
from copy import copy

from clinica.utils.inputs import fetch_file, RemoteFileStructure
from .utils import im_loss_roi_gaussian_distribution, find_image_path, load_and_check_tsv, generate_shepplogan_phantom
from ..tsv.tsv_utils import extract_baseline
from ..deep_learning.data import MRIDataset
from clinicadl.tools.inputs.filename_types import FILENAME_TYPE
from clinicadl.tools.deep_learning.iotools import check_and_clean, commandline_to_json
import tarfile


def generate_random_dataset(caps_dir, output_dir, n_subjects, tsv_path=None, mean=0,
                            sigma=0.5, preprocessing="t1-linear", multi_cohort=False):
    """
    Generates a random dataset.

    Creates a random dataset for intractable classification task from the first
    subject of the tsv file (other subjects/sessions different from the first
    one are ignored. Degree of noise can be parameterized.

    Args:
        caps_dir: (str) Path to the (input) CAPS directory.
        output_dir: (str) folder containing the synthetic dataset in (output)
            CAPS format.
        n_subjects: (int) number of subjects in each class of the
            synthetic dataset
        tsv_path: (str) path to tsv file of list of subjects/sessions.
        mean: (float) mean of the gaussian noise
        sigma: (float) standard deviation of the gaussian noise
        preprocessing: (str) preprocessing performed. Must be in ['t1-linear', 't1-extensive'].
        multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.

    Returns:
        A folder written on the output_dir location (in CAPS format), also a
        tsv file describing this output

    """
    commandline_to_json({
        "output_dir": output_dir,
        "caps_dir": caps_dir,
        "preprocessing": preprocessing,
        "n_subjects": n_subjects,
        "mean": mean,
        "sigma": sigma
    })
    # Transform caps_dir in dict
    caps_dict = MRIDataset.create_caps_dict(caps_dir, multi_cohort=multi_cohort)

    # Read DataFrame
    data_df = load_and_check_tsv(tsv_path, caps_dict, output_dir)

    # Create subjects dir
    makedirs(join(output_dir, 'subjects'), exist_ok=True)

    # Retrieve image of first subject
    participant_id = data_df.loc[0, 'participant_id']
    session_id = data_df.loc[0, 'session_id']
    cohort = data_df.loc[0, 'cohort']

    image_path = find_image_path(caps_dict, participant_id, session_id, cohort, preprocessing)
    image_nii = nib.load(image_path)
    image = image_nii.get_data()

    # Create output tsv file
    participant_id_list = [f'sub-RAND{i}' for i in range(2 * n_subjects)]
    session_id_list = ['ses-M00'] * 2 * n_subjects
    diagnosis_list = ['AD'] * n_subjects + ['CN'] * n_subjects
    data = np.array([participant_id_list, session_id_list, diagnosis_list])
    data = data.T
    output_df = pd.DataFrame(data, columns=['participant_id', 'session_id', 'diagnosis'])
    output_df['age_bl'] = 60
    output_df['sex'] = 'F'
    output_df.to_csv(join(output_dir, 'data.tsv'), sep='\t', index=False)

    for i in range(2 * n_subjects):
        gauss = np.random.normal(mean, sigma, image.shape)
        participant_id = f'sub-RAND{i}'
        noisy_image = image + gauss
        noisy_image_nii = nib.Nifti1Image(noisy_image, header=image_nii.header, affine=image_nii.affine)
        noisy_image_nii_path = join(output_dir, 'subjects', participant_id, 'ses-M00', 't1_linear')
        noisy_image_nii_filename = participant_id + '_ses-M00' + FILENAME_TYPE['cropped'] + '.nii.gz'
        makedirs(noisy_image_nii_path, exist_ok=True)
        nib.save(noisy_image_nii, join(noisy_image_nii_path, noisy_image_nii_filename))

    missing_path = join(output_dir, "missing_mods")
    makedirs(missing_path, exist_ok=True)

    sessions = output_df.session_id.unique()
    for session in sessions:
        session_df = output_df[output_df.session_id == session]
        out_df = copy(session_df[["participant_id"]])
        out_df["synthetic"] = [1] * len(out_df)
        out_df.to_csv(join(missing_path, f"missing_mods_{session}.tsv"), sep="\t", index=False)


def generate_trivial_dataset(caps_dir, output_dir, n_subjects, tsv_path=None, preprocessing="linear",
                             mask_path=None, atrophy_percent=60, multi_cohort=False):
    """
    Generates a fully separable dataset.

    Generates a dataset, based on the images of the CAPS directory, where a
    half of the image is processed using a mask to oclude a specific region.
    This procedure creates a dataset fully separable (images with half-right
    processed and image with half-left processed)

    Args:
        caps_dir: (str) path to the CAPS directory.
        output_dir: (str) folder containing the synthetic dataset in CAPS format.
        n_subjects: (int) number of subjects in each class of the synthetic
            dataset.
        tsv_path: (str) path to tsv file of list of subjects/sessions.
        preprocessing: (str) preprocessing performed. Must be in ['linear', 'extensive'].
        mask_path: (str) path to the extracted masks to generate the two labels.
        atrophy_percent: (float) percentage of atrophy applied.
        multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.

    Returns:
        Folder structure where images are stored in CAPS format.

    Raises:
        ValueError: if `n_subjects` is higher than the length of the TSV file at `tsv_path`.
    """
    from pathlib import Path

    commandline_to_json({
        "output_dir": output_dir,
        "caps_dir": caps_dir,
        "preprocessing": preprocessing,
        "n_subjects": n_subjects,
        "atrophy_percent": atrophy_percent
    })

    # Transform caps_dir in dict
    caps_dict = MRIDataset.create_caps_dict(caps_dir, multi_cohort=multi_cohort)

    # Read DataFrame
    data_df = load_and_check_tsv(tsv_path, caps_dict, output_dir)
    data_df = extract_baseline(data_df, "None")

    home = str(Path.home())
    cache_clinicadl = join(home, '.cache', 'clinicadl', 'ressources', 'masks')
    url_aramis = 'https://aramislab.paris.inria.fr/files/data/masks/'
    FILE1 = RemoteFileStructure(filename='AAL2.tar.gz',
                                url=url_aramis,
                                checksum='89427970921674792481bffd2de095c8fbf49509d615e7e09e4bc6f0e0564471'
                                )
    makedirs(cache_clinicadl, exist_ok=True)

    if n_subjects > len(data_df):
        raise ValueError(f"The number of subjects {n_subjects} cannot be higher "
                         f"than the number of subjects in the baseline dataset of size {len(data_df)}")

    if mask_path is None:
        if not exists(join(cache_clinicadl, 'AAL2')):
            try:
                print('Try to download AAL2 masks')
                mask_path_tar = fetch_file(FILE1, cache_clinicadl)
                tar_file = tarfile.open(mask_path_tar)
                print('File: ' + mask_path_tar)
                try:
                    tar_file.extractall(cache_clinicadl)
                    tar_file.close()
                    mask_path = join(cache_clinicadl, 'AAL2')
                except RuntimeError:
                    print('Unable to extract downloaded files')
            except IOError as err:
                print('Unable to download required templates:', err)
                raise ValueError('''Unable to download masks, please download them
                                  manually at https://aramislab.paris.inria.fr/files/data/masks/
                                  and provide a valid path.''')
        else:
            mask_path = join(cache_clinicadl, 'AAL2')

    # Create subjects dir
    makedirs(join(output_dir, 'subjects'), exist_ok=True)

    # Output tsv file
    columns = ['participant_id', 'session_id', 'diagnosis', 'age_bl', 'sex']
    output_df = pd.DataFrame(columns=columns)
    diagnosis_list = ["AD", "CN"]

    for i in range(2 * n_subjects):
        data_idx = i // 2
        label = i % 2

        participant_id = data_df.loc[data_idx, "participant_id"]
        session_id = data_df.loc[data_idx, "session_id"]
        cohort = data_df.loc[data_idx, "cohort"]
        filename = f'sub-TRIV{i}_ses-M00' + FILENAME_TYPE['cropped'] + '.nii.gz'
        path_image = join(output_dir, 'subjects', f'sub-TRIV{i}', 'ses-M00', 't1_linear')

        makedirs(path_image, exist_ok=True)

        image_path = find_image_path(caps_dict, participant_id, session_id, cohort, preprocessing)
        image_nii = nib.load(image_path)
        image = image_nii.get_data()

        atlas_to_mask = nib.load(join(mask_path, f'mask-{label + 1}.nii')).get_data()

        # Create atrophied image
        trivial_image = im_loss_roi_gaussian_distribution(image, atlas_to_mask, atrophy_percent)
        trivial_image_nii = nib.Nifti1Image(trivial_image, affine=image_nii.affine)
        trivial_image_nii.to_filename(join(path_image, filename))

        # Append row to output tsv
        row = [f'sub-TRIV{i}', 'ses-M00', diagnosis_list[label], 60, 'F']
        row_df = pd.DataFrame([row], columns=columns)
        output_df = output_df.append(row_df)

    output_df.to_csv(join(output_dir, 'data.tsv'), sep='\t', index=False)

    missing_path = join(output_dir, "missing_mods")
    makedirs(missing_path, exist_ok=True)

    sessions = output_df.session_id.unique()
    for session in sessions:
        session_df = output_df[output_df.session_id == session]
        out_df = copy(session_df[["participant_id"]])
        out_df["synthetic"] = [1] * len(out_df)
        out_df.to_csv(join(missing_path, f"missing_mods_{session}.tsv"), sep="\t", index=False)


def generate_shepplogan_dataset(output_dir, img_size, labels_distribution,
                                samples=100, smoothing=True):

    check_and_clean(join(output_dir, "subjects"))
    commandline_to_json({
        "output_dir": output_dir,
        "img_size": img_size,
        "labels_distribution": labels_distribution,
        "samples": samples,
        "smoothing": smoothing
    })
    columns = ["participant_id", "session_id", "diagnosis", "subtype"]
    data_df = pd.DataFrame(columns=columns)

    for i, label in enumerate(labels_distribution.keys()):
        samples_per_subtype = np.array(labels_distribution[label]) * samples
        for subtype in range(len(samples_per_subtype)):
            for j in range(int(samples_per_subtype[subtype])):
                participant_id = "sub-CLNC%i%04d" % (i, j + np.sum(samples_per_subtype[:subtype:]).astype(int))
                session_id = "ses-M00"
                row_df = pd.DataFrame([[participant_id, session_id, label, subtype]],
                                      columns=columns)
                data_df = data_df.append(row_df)

                # Image generation
                path_out = join(output_dir, "subjects", "%s_%s%s.pt"
                                % (participant_id, session_id, FILENAME_TYPE["shepplogan"]))
                img = generate_shepplogan_phantom(img_size, label=subtype, smoothing=smoothing)
                torch_img = torch.from_numpy(img).float().unsqueeze(0)
                torch.save(torch_img, path_out)

    data_df.to_csv(join(output_dir, 'data.tsv'), sep="\t", index=False)

    missing_path = join(output_dir, "missing_mods")
    if not exists(missing_path):
        makedirs(missing_path)

    sessions = data_df.session_id.unique()
    for session in sessions:
        session_df = data_df[data_df.session_id == session]
        out_df = copy(session_df[["participant_id"]])
        out_df["t1w"] = [1] * len(out_df)
        out_df.to_csv(join(missing_path, "missing_mods_%s.tsv" % session), sep="\t", index=False)


def generate_shepplogan_dataset(output_dir, img_size, labels_distribution,
                                samples=100, smoothing=True):

    check_and_clean(join(output_dir, "subjects"))
    commandline_to_json({
        "output_dir": output_dir,
        "img_size": img_size,
        "labels_distribution": labels_distribution,
        "samples": samples,
        "smoothing": smoothing
    })
    columns = ["participant_id", "session_id", "diagnosis", "subtype"]
    data_df = pd.DataFrame(columns=columns)

    for i, label in enumerate(labels_distribution.keys()):
        for j in range(samples):
            participant_id = "sub-CLNC%i%04d" % (i, j)
            session_id = "ses-M00"
            subtype = np.random.choice(np.arange(len(labels_distribution[label])), p=labels_distribution[label])
            row_df = pd.DataFrame([[participant_id, session_id, label, subtype]],
                                  columns=columns)
            data_df = data_df.append(row_df)

            # Image generation
            path_out = join(output_dir, "subjects", "%s_%s%s.pt"
                            % (participant_id, session_id, FILENAME_TYPE["shepplogan"]))
            img = generate_shepplogan_phantom(img_size, label=subtype, smoothing=smoothing)
            torch_img = torch.from_numpy(img).float().unsqueeze(0)
            torch.save(torch_img, path_out)

    data_df.to_csv(join(output_dir, 'data.tsv'), sep="\t", index=False)

    missing_path = join(output_dir, "missing_mods")
    if not exists(missing_path):
        makedirs(missing_path)

    sessions = data_df.session_id.unique()
    for session in sessions:
        session_df = data_df[data_df.session_id == session]
        out_df = copy(session_df[["participant_id"]])
        out_df["t1w"] = [1] * len(out_df)
        out_df.to_csv(join(missing_path, "missing_mods_%s.tsv" % session), sep="\t", index=False)
