"""
Produces a tsv file to study all the nii files and perform the quality check.
"""
import os
from os import path
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from clinica.utils.inputs import RemoteFileStructure, fetch_file


def distance(mni_mask_np, img_np):

    shape3D = img_np.shape

    if not (img_np.shape == mni_mask_np.shape):
        print("numpy array hasn't the same size and cannot be compare")

    sum_in_contour = 0
    sum_out_contour = 0
    nb_contour = 1605780
    distance_ = 0

    for i in range(shape3D[0]):
        for j in range(shape3D[1]):
            for k in range(shape3D[2]):

                tmp = img_np[i][j][k]
                tmp_mni = int(mni_mask_np[i][j][k])

                if tmp_mni == 0:
                    sum_out_contour += tmp

                elif tmp_mni == 1:
                    if 0.2 < tmp < 0.8:
                        sum_in_contour += tmp

                diff_ = abs(tmp_mni - tmp)

                distance_ += diff_

    return distance_, sum_in_contour / nb_contour


def extract_metrics(caps_dir, output_dir, acq_label, ref_region, threshold):
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # NEED TO ADD THE PATH TO THE MNI MASK OR TO THE MNI IMAGE
    # Load eyes segmentation
    home = str(Path.home())
    cache_clinicadl = path.join(home, ".cache", "clinicadl", "mask")
    url_aramis = "https://aramislab.paris.inria.fr/files/data/template/"
    FILE1 = RemoteFileStructure(
        filename="mask_contour.nii.gz",
        url=url_aramis,
        checksum="56f699c06cafc62ad8bb5b41b188c7c412d684d810a11d6f4cbb441c0ce944ee",
    )

    FILE2 = RemoteFileStructure(
        filename="mask_brain.nii.gz",
        url=url_aramis,
        checksum="56f699c06cafc62ad8bb5b41b188c7c412d684d810a11d6f4cbb441c0ce944ee",
    )

    if not (path.exists(cache_clinicadl)):
        os.makedirs(cache_clinicadl)

    mask_contour_file = path.join(cache_clinicadl, FILE1.filename)

    if not (path.exists(mask_contour_file)):
        try:
            mask_contour_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            raise IOError("Unable to download required mni file for QC:", err)

    mni_mask_nii = nib.load(mask_contour_file)
    mni_mask_np = mni_mask_nii.get_fdata()

    # Get the data
    filename = path.join(output_dir, "QC_metrics.tsv")
    columns = [
        "participant_id",
        "session_id",
        "max_intensity",
        "ttp_ar",
        "tfp_av",
    ]
    results_df = pd.DataFrame()

    subjects = os.listdir(path.join(caps_dir, "subjects"))
    subjects = [subject for subject in subjects if subject[:4:] == "sub-"]
    for subject in subjects:
        subject_path = path.join(caps_dir, "subjects", subject)
        sessions = os.listdir(subject_path)
        sessions = [session for session in sessions if session[:4:] == "ses-"]
        for session in sessions:
            image_path = path.join(
                subject_path,
                session,
                "pet_linear",
                subject
                + "_"
                + session
                + "trc-"
                + acq_label
                + "_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-"
                + ref_region
                + "_pet.nii.gz",
            )
            tensor_path = path.join(
                subject_path,
                session,
                "deeplearning_prepare_data",
                "image_based",
                "pet_linear",
                subject
                + "_"
                + session
                + "_trc-"
                + acq_label
                + "_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-"
                + ref_region
                + "_pet.pt",
            )

            if path.exists(tensor_path):
                image_torch = torch.load(tensor_path)
                image_np = image_torch.numpy()
            elif path.exists(image_path):
                image_nii = nib.load(image_path)
                image_np = image_nii.get_fdata()
            else:
                raise FileNotFoundError(f"Clinical data not found ({image_path})")

            ttp_av, tfp_arr = distance(mni_mask_np, image_np)

            row = [
                [
                    subject,
                    session,
                    np.max(image_np),
                    ttp_av,
                    tfp_arr,
                ]
            ]
            row_df = pd.DataFrame(row, columns=columns)
            results_df = pd.concat([results_df, row_df])

    results_df.sort_values("max_intensity", inplace=True, ascending=True)
    results_df.to_csv(filename, sep="\t", index=False)
