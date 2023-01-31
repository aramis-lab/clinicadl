"""
Automatically reject images incorrectly preprocessed pet-linear (Unified Segmentation) with ? criterion


"""
import os
from logging import getLogger
from os import path
from pathlib import Path

import nibabel as nib
import pandas as pd
from clinica.utils.inputs import RemoteFileStructure, fetch_file
from joblib import Parallel, delayed

from .utils import distance, extract_mask


def quality_check(
    caps_dir: str,
    output_tsv: str,
    acq_label: str,
    ref_region: str,
    participants_tsv: str = None,
    threshold: float = 0.9,
    n_proc: int = 0,
    gpu: bool = False,
):

    logger = getLogger("clinicadl.quality_check")

    if path.exists(output_tsv):
        raise NameError("this file already exists please chose another name")

    # NEED TO ADD THE PATH TO THE MNI MASK OR TO THE MNI IMAGE
    # Load eyes segmentation
    home = str(Path.home())
    cache_clinicadl = path.join(home, ".cache", "clinicadl", "mask")
    if not (path.exists(cache_clinicadl)):
        os.makedirs(cache_clinicadl)

    url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/"
    FILE1 = RemoteFileStructure(
        filename="qc_pet_mask_contour.nii.gz",
        url=url_aramis,
        checksum="0c561ce7de343219e42861b87a359420f9d485da37a8f64d1366ee9bb5460ee6",
    )

    mask_contour_file = path.join(cache_clinicadl, FILE1.filename)
    if not (path.exists(mask_contour_file)):
        try:
            mask_contour_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            raise IOError("Unable to download required mni file for QC:", err)

    mask_contour_nii = nib.load(mask_contour_file)
    mask_contour = mask_contour_nii.get_fdata()
    mask_contour.astype(int)

    FILE2 = RemoteFileStructure(
        filename="qc_pet_mask_brain.nii.gz",
        url=url_aramis,
        checksum="e78a542da49755f5c9ba751b4acca725650396999a671831f0acd8fbf4b898e8",
    )

    mask_brain_file = path.join(cache_clinicadl, FILE2.filename)
    if not (path.exists(mask_brain_file)):
        try:
            mask_brain_file = fetch_file(FILE2, cache_clinicadl)
        except IOError as err:
            raise IOError("Unable to download required mni file for QC:", err)

    mask_brain_nii = nib.load(mask_brain_file)
    mask_brain = mask_brain_nii.get_fdata()
    mask_brain.astype(int)

    # Get the data
    columns = [
        "participant_id",
        "session_id",
        "tfp_brain",
        "mttp_contour",
        "sum_contour_35",
        "sum_brain_35",
    ]

    results_df = pd.DataFrame(columns=columns)

    if not participants_tsv is None:
        participants_df = pd.read_csv(participants_tsv, sep="\t")
        participants_df.reset_index()
        participants_df.set_index(["participant_id", "session_id"], inplace=True)
        subjects = [subject for subject, subject_df in participants_df.groupby(level=0)]
    else:
        subjects = os.listdir(path.join(caps_dir, "subjects"))
        subjects = [subject for subject in subjects if subject[:4:] == "sub-"]

    def parallelize_subjects(subject, results_df):
        subject_path = path.join(caps_dir, "subjects", subject)

        if not participants_tsv is None:
            sessions = participants_df.loc[subject]
            sessions.reset_index(inplace=True)
            sessions = sessions["session_id"].to_list()
        else:
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

            if path.exists(image_path):
                image_nii = nib.load(image_path)
                image_np = image_nii.get_fdata()
            else:
                raise FileNotFoundError(f"Clinical data not found ({image_path})")

            tfp_brain, mttp_contour, sum_contour_35, sum_brain_35 = distance(
                mask_contour, mask_brain, image_np
            )

            row = [
                [
                    subject,
                    session,
                    tfp_brain,
                    mttp_contour,
                    sum_contour_35,
                    sum_brain_35,
                ]
            ]
            row_df = pd.DataFrame(row, columns=columns)
            results_df = pd.concat([results_df, row_df], ignore_index=True)
        return results_df

    results_df = Parallel(n_jobs=n_proc)(
        delayed(parallelize_subjects)(subject, results_df) for subject in subjects
    )

    all_df = pd.DataFrame(columns=columns)
    for subject_df in results_df:
        all_df = pd.concat([all_df, subject_df])
    all_df.sort_values("sum_contour_35", inplace=True, ascending=True)
    all_df.to_csv(output_tsv, sep="\t", index=False)

    logger.info(
        f"Quality check metrics extracted at {path.join(output_tsv, 'QC_metrics.tsv')}."
    )
