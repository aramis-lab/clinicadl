"""
Automatically reject images incorrectly preprocessed pet-linear (Unified Segmentation) with ? criterion


"""

from logging import getLogger
from pathlib import Path

import nibabel as nib
import pandas as pd
from clinica.utils.inputs import RemoteFileStructure, fetch_file
from joblib import Parallel, delayed

from .utils import distance


def quality_check(
    caps_dir: str,
    output_tsv: str,
    acq_label: str,
    ref_region: str,
    participants_tsv: str = None,
    threshold: float = 0.8,
    n_proc: int = 0,
    gpu: bool = False,
):
    """
    Performs quality check on pet-linear pipeline.

    Parameters
    ----------

    caps_directory: str (Path)
        The CAPS folder where pet-linear outputs are stored.
    output_tsv: str (Path)
        The path to TSV output file.
    acq_label: str
        The label given to the PET acquisition, specifying the tracer used (trc-<acq_label>).
    ref_region: str
        The reference region used to perform intensity normalization {pons|cerebellumPons|pons2|cerebellumPons2}.
    participants_tsv: str (Path)
        Path to a TSV file including a list of participants/sessions on which the quality-check will be performed.
    threshold: float
        The threshold on the output probability to decide if the image passed or failed.
        Default is 0.8
    n_proc: int
        Number of cores used during the task.
    """

    logger = getLogger("clinicadl.quality_check")

    if Path(output_tsv).is_file():
        raise NameError("this file already exists please chose another name")

    # load the contour mask
    home = Path.home()
    cache_clinicadl = home / ".cache" / "clinicadl" / "mask"
    if not cache_clinicadl.is_dir():
        cache_clinicadl.mkdir(parents=True)

    url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/"
    FILE1 = RemoteFileStructure(
        filename="qc_pet_mask_contour.nii.gz",
        url=url_aramis,
        checksum="0c561ce7de343219e42861b87a359420f9d485da37a8f64d1366ee9bb5460ee6",
    )

    mask_contour_file = cache_clinicadl / FILE1.filename

    if not (mask_contour_file).is_file():
        try:
            mask_contour_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            raise IOError("Unable to download required MNI file for QC: ", err)

    mask_contour_nii = nib.load(mask_contour_file)
    mask_contour = mask_contour_nii.get_fdata()
    mask_contour.astype(int)

    # Get the data
    columns = [
        "participant_id",
        "session_id",
        "pass_probability",
        "pass",
    ]

    results_df = pd.DataFrame(columns=columns)

    if not participants_tsv is None:
        participants_df = pd.read_csv(participants_tsv, sep="\t")
        participants_df.reset_index()
        participants_df.set_index(["participant_id", "session_id"], inplace=True)
        subjects = [subject for subject, subject_df in participants_df.groupby(level=0)]
    else:
        subjects = list((Path(caps_dir) / "subjects").iterdir())
        subjects = [subject.name for subject in subjects if subject.name[:4:] == "sub-"]

    def parallelize_subjects(subject, results_df):
        subject_path = Path(caps_dir) / "subjects" / subject

        if not participants_tsv is None:
            sessions = participants_df.loc[subject]
            sessions.reset_index(inplace=True)
            sessions = sessions["session_id"].to_list()
        else:
            sessions = list(subject_path.iterdir())
            sessions = [
                session.name for session in sessions if session.name[:4:] == "ses-"
            ]

        for session in sessions:
            image_path = (
                subject_path
                / session
                / "pet_linear"
                / (
                    subject
                    + "_"
                    + session
                    + "_trc-"
                    + acq_label
                    + "_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-"
                    + ref_region
                    + "_pet.nii.gz"
                )
            )

            if image_path.is_file():
                image_nii = nib.load(image_path)
                image_np = image_nii.get_fdata()
            else:
                raise FileNotFoundError(f"Clinical data not found ({image_path})")

            sum_contour_35 = distance(mask_contour, image_np)

            row = [
                [
                    subject,
                    session,
                    sum_contour_35,
                    sum_contour_35 < threshold,
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
    col_tmp = all_df["pass_probability"]
    all_df["pass_probability"] = 1 - (
        (col_tmp - col_tmp.min()) / (col_tmp.max() - col_tmp.min())
    )
    all_df["pass"] = all_df["pass_probability"] < threshold
    all_df.sort_values("pass_probability", inplace=True, ascending=True)
    all_df.to_csv(output_tsv, sep="\t", index=False)

    logger.info(
        f"Quality check metrics extracted at {Path(output_tsv) / 'QC_metrics.tsv'}."
    )
