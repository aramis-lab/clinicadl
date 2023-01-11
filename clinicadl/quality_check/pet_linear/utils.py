"""
Produces a tsv file to study all the nii files and perform the quality check.
"""
import os
from os import path
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from clinica.utils.inputs import RemoteFileStructure, fetch_file
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu


def thresholdOtsuDisplay(image):
    thresh = threshold_otsu(image[np.isfinite(image)])
    binary = image > thresh
    return binary


def normalize_homemade(image_np):
    max_base = image_np.max()
    min_base = image_np.min()
    for i in range(len(image_np)):
        for j in range(len(image_np[i])):
            for k in range(len(image_np[i][j])):

                image_np[i][j][k] = (image_np[i][j][k] - min_base) / (
                    max_base - min_base
                )
    return image_np


def extract_mask(img_np):

    image_np = normalize_homemade(img_np)

    kernel = np.ones((3, 3), np.uint8)

    mask = thresholdOtsuDisplay(image_np)
    # print(mask.astype("uint8"))
    mask = cv2.dilate(mask.astype("uint8"), kernel, iterations=4)

    mask = cv2.erode(mask.astype("uint8"), kernel, iterations=5)

    mask = binary_fill_holes(mask)

    return mask


def distance(mask_contour, mask_brain, mask_np):

    shape3D = mask_np.shape

    if not (shape3D == mask_brain.shape):
        print("numpy array hasn't the same size and cannot be compare")

    tn_contour_av = 0
    tp_contour_av = 0
    fp_contour_av = 0
    fn_contour_av = 0
    print("test")
    tn_brain_arr = 0
    tp_brain_arr = 0
    fp_brain_arr = 0
    fn_brain_arr = 0

    tfp_brain_arr = 0
    mttp_contour_av = 0
    nb_contour_av = 373921
    nb_brain_arr = 616879

    xx = int(shape3D[0] / 2)
    yy = shape3D[1]
    yy_limit = 103
    zz = shape3D[2]

    for i in range(xx):
        for j in range(yy):
            for k in range(zz):

                tmp = mask_np[i][j][k]
                tmp_brain = int(mask_brain[i][j][k])
                tmp_contour = int(mask_contour[i][j][k])

                if j <= yy_limit:
                    if tmp_brain == 0:
                        if tmp == 0:
                            tn_brain_arr += 1
                        if tmp == 1:
                            fn_brain_arr += 1

                    elif tmp_brain == 1:
                        if tmp == 0:
                            tp_brain_arr += 1
                        elif tmp == 1:
                            fp_brain_arr += 1

                elif j > yy_limit:
                    if tmp_contour == 0:
                        if tmp == 0:
                            tn_contour_av += 1
                        elif tmp == 1:
                            fn_contour_av += 1

                    elif tmp_contour == 1:
                        if tmp == 0:
                            fp_contour_av += 1
                        elif tmp == 1:
                            tp_contour_av += 1

    tfp_brain_arr = tp_brain_arr / (fp_brain_arr + tp_brain_arr)
    mttp_contour_av = 1 - (fp_contour_av / (fp_contour_av + tp_contour_av))
    return tfp_brain_arr, mttp_contour_av


def extract_metrics(caps_dir, output_dir, acq_label, ref_region):
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    # NEED TO ADD THE PATH TO THE MNI MASK OR TO THE MNI IMAGE
    # Load eyes segmentation
    # home = str(Path.home())
    # cache_clinicadl = path.join(home, ".cache", "clinicadl", "mask")
    # if not (path.exists(cache_clinicadl)):
    #     os.makedirs(cache_clinicadl)

    # url_aramis = "/Users/camille.brianceau/aramis/QC_tokeep/mask"
    # FILE1 = RemoteFileStructure(
    #     filename="contour.nii.gz",
    #     url=url_aramis,
    #     checksum="56f699c06cafc62ad8bb5b41b188c7c412d684d810a11d6f4cbb441c0ce944ee",
    # )

    # mask_contour_file = path.join(cache_clinicadl, FILE1.filename)
    # if not (path.exists(mask_contour_file)):
    #     try:
    #         mask_contour_file = fetch_file(FILE1, cache_clinicadl)
    #     except IOError as err:
    #         raise IOError("Unable to download required mni file for QC:", err)

    mask_contour_nii = nib.load(
        "/Users/camille.brianceau/aramis/QC_tokeep/mask/contour.nii.gz"
    )
    mask_contour = mask_contour_nii.get_fdata()
    print("test 2")
    # FILE2 = RemoteFileStructure(
    #     filename="mask_brain.nii.gz",
    #     url=url_aramis,
    #     checksum="56f699c06cafc62ad8bb5b41b188c7c412d684d810a11d6f4cbb441c0ce944ee",
    # )

    # mask_brain_file = path.join(cache_clinicadl, FILE1.filename)

    # if not (path.exists(mask_brain_file)):
    #     try:
    #         mask_brain_file = fetch_file(FILE1, cache_clinicadl)
    #     except IOError as err:
    #         raise IOError("Unable to download required mni file for QC:", err)

    mask_brain_nii = nib.load(
        "/Users/camille.brianceau/aramis/QC_tokeep/mask/test_brain_dilate.nii.gz"
    )
    mask_brain = mask_brain_nii.get_fdata()

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
                + "_trc-"
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
                image_np_torch = image_torch.numpy()
                print(tensor_path)
            if path.exists(image_path):
                image_nii = nib.load(image_path)
                image_np = image_nii.get_fdata()
                print(image_path)
            else:
                raise FileNotFoundError(f"Clinical data not found ({image_path})")
            comparison = image_np_torch == image_np
            equal_arrays = comparison.all()

            print(equal_arrays)
            print(type(image_np))
            print(type(image_np_torch))
            tfp_brain_arr, mttp_contour_av = distance(
                mask_contour, mask_brain, extract_mask(image_np)
            )

            row = [
                [
                    subject,
                    session,
                    np.max(image_np),
                    mttp_contour_av,
                    tfp_brain_arr,
                ]
            ]
            row_df = pd.DataFrame(row, columns=columns)
            results_df = pd.concat([results_df, row_df])

    results_df.sort_values("max_intensity", inplace=True, ascending=True)
    results_df.to_csv(filename, sep="\t", index=False)
