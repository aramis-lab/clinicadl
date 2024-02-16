# coding: utf8

"""
Source files can be obtained by running the following commands on a BIDS folder:
 - clinica iotools merge-tsv
 - clinica iotools check-missing-modalities
To download Clinica follow the instructions at http://www.clinica.run/doc/#installation

NB: Other preprocessing may be needed on the merged file obtained: for example the selection of subjects older than 62
in the OASIS dataset is not done in this script. Moreover a quality check may be needed at the end of preprocessing
pipelines, leading to the removal of some subjects.
"""
from copy import copy
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import (
    cleaning_nan_diagnoses,
    find_label,
    first_session,
    last_session,
    neighbour_session,
)

logger = getLogger("clinicadl.tsvtools")


def infer_or_drop_diagnosis(bids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduce the diagnosis when missing from previous and following sessions of the subject. If not identical, the session
    is dropped. Sessions with no diagnosis are also dropped when there are the last sessions of the follow-up.

    Parameters
    ----------
    bids_df: DataFrame
        Columns including ['participant_id', 'session_id', 'diagnosis'].

    Returns
    -------
    bids_copy_df: DataFrame
        Cleaned copy of the input bids_df.
    """
    bids_copy_df = copy(bids_df)
    found_diag_interpol = 0
    nb_drop = 0

    for subject, subject_df in bids_df.groupby(level=0):
        session_list = []
        for _, session in subject_df.index.values:
            x = session[5::]
            if not x.isdigit():
                subject_df.drop((subject, session), axis=0, inplace=True)
                bids_copy_df.drop((subject, session), axis=0, inplace=True)
                nb_drop += 1

        session_list = [session for _, session in subject_df.index.values]

        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), "diagnosis"]
            session_nb = session

            if isinstance(diagnosis, float):
                if session == last_session(session_list):
                    bids_copy_df.drop(index=(_, session), axis=0, inplace=True)
                    nb_drop += 1
                else:
                    prev_session = neighbour_session(session_nb, session_list, -1)
                    prev_diagnosis = bids_df.loc[(subject, prev_session), "diagnosis"]
                    while isinstance(
                        prev_diagnosis, float
                    ) and prev_session != first_session(subject_df):
                        prev_session = neighbour_session(prev_session, session_list, -1)
                        prev_diagnosis = bids_df.loc[
                            (subject, prev_session), "diagnosis"
                        ]
                    post_session = neighbour_session(session_nb, session_list, +1)
                    post_diagnosis = bids_df.loc[(subject, post_session), "diagnosis"]
                    while isinstance(
                        post_diagnosis, float
                    ) and post_session != last_session(session_list):
                        post_session = neighbour_session(post_session, session_list, +1)
                        post_diagnosis = bids_df.loc[
                            (subject, post_session), "diagnosis"
                        ]
                    if prev_diagnosis == post_diagnosis:
                        found_diag_interpol += 1
                        bids_copy_df.loc[(subject, session), "diagnosis"] = (
                            prev_diagnosis
                        )
                    else:
                        bids_copy_df.drop((subject, session), inplace=True)
                        nb_drop += 1

    logger.info(f"Inferred diagnosis: {found_diag_interpol}")
    logger.info(f"Dropped subjects (inferred diagnosis): {nb_drop}")

    return bids_copy_df


def mod_selection(
    bids_df: pd.DataFrame, missing_mods_dict: Dict[str, pd.DataFrame], mod: str = "t1w"
) -> pd.DataFrame:
    """
    Select only sessions for which the modality is present

    Parameters
    ----------
    bids_df: DataFrame
        Columns include ['participant_id', 'session_id', 'diagnosis']
    missing_mods_dict: dictionary of str and DataFrame
        DataFrames of missing modalities
    mod: str
        the modality used for selection

    Returns
    -------
    copy_bids_df: DataFrame
        Cleaned copy of the input bids_df
    """
    bids_copy_df = copy(bids_df)
    nb_subjects = 0
    if mod is not None:
        for subject, session in bids_df.index.values:
            try:
                mod_present = missing_mods_dict[session].loc[subject, mod]
                if not mod_present:
                    bids_copy_df.drop((subject, session), inplace=True)
                    nb_subjects += 1
            except KeyError:
                bids_copy_df.drop((subject, session), inplace=True)
                nb_subjects += 1
    logger.info(f"Dropped sessions (mod selection): {nb_subjects}")
    return bids_copy_df


def remove_unique_session(bids_df: pd.DataFrame) -> pd.DataFrame:
    """
    A method to get the subgroup for each sessions depending on their stability on the time horizon

    Parameters
    ----------
    bids_df: DataFrame
        Columns include ['participant_id', 'session_id', 'diagnosis']

    Returns
    -------
    bids_copy_df: DataFrame
        Cleaned copy of the input bids_df
    """
    bids_copy_df = copy(bids_df)
    nb_unique = 0

    for subject, subject_df in bids_df.groupby(level=0):
        session_list = [session for _, session in subject_df.index.values]
        session_list.sort()
        nb_session = len(session_list)
        if nb_session == 1:
            bids_copy_df.drop((subject, session_list[0]), inplace=True)
            subject_df.drop((subject, session_list[0]), inplace=True)
            nb_unique += 1
    logger.info(f"Dropped subjects (unique session): {nb_unique}")

    return bids_copy_df


def diagnosis_removal(bids_df: pd.DataFrame, diagnosis_list: List[str]) -> pd.DataFrame:
    """
    Removes sessions for which the diagnosis is not in the list provided

    Parameters
    ----------
    bids_df: DataFrame
        Columns must includes ['participant_id', 'session_id', 'diagnosis']
    diagnosis_list: list of str
        List of diagnoses that will be removed

    Returns
    -------
    output_df: DataFrame
        Cleaned copy of the input bids_df

    """

    output_df = copy(bids_df)
    nb_subjects = 0
    for subject, subject_df in bids_df.groupby(level=0):
        for _, session in subject_df.index.values:
            group = subject_df.loc[(subject, session), "diagnosis"]
            if group not in diagnosis_list:
                output_df.drop((subject, session), inplace=True)
                nb_subjects += 1

    logger.info(f"Dropped subjects (diagnoses): {nb_subjects}")
    return output_df


def apply_restriction(bids_df: pd.DataFrame, restriction_path: Path) -> pd.DataFrame:
    """
    Application of a restriction (for example after the removal of some subjects after a preprocessing pipeline)

    Parameters
    ----------
    bids_df: DataFrame
        Columns must include ['participant_id', 'session_id', 'diagnosis']
    restriction_path: str (path)
        Path to a tsv file with columns including ['participant_id', 'session_id', 'diagnosis'] including
        all the sessions that can be included

    Returns
    -------
    bids_copy_df: DataFrame
        Cleaned copy of the input bids_df
    """
    bids_copy_df = copy(bids_df)
    nb_subjects = 0
    if restriction_path is not None:
        restriction_df = pd.read_csv(restriction_path, sep="\t")

        for subject, session in bids_df.index.values:
            subject_qc_df = restriction_df[
                (restriction_df.participant_id == subject)
                & (restriction_df.session_id == session)
            ]
            if len(subject_qc_df) != 1:
                bids_copy_df.drop((subject, session), inplace=True)
                nb_subjects += 1
    logger.info(f"Dropped subjects (apply restriction): {nb_subjects}")
    return bids_copy_df


def get_labels(
    bids_directory: Path,
    diagnoses: List[str],
    modality: str = "t1w",
    restriction_path: Path = None,
    variables_of_interest: List[str] = None,
    remove_smc: bool = True,
    merged_tsv: Path = None,
    missing_mods: Path = None,
    remove_unique_session_: bool = False,
    output_dir: Path = None,
    caps_directory: Path = None,
):
    """
    Writes one TSV file based on merged_tsv and missing_mods.


    Parameters
    ----------
    bids_directory: str (path)
        Path to the folder containing the dataset in a BIDS hierarchy.
    diagnoses: List of str
        Labels that must be extracted from merged_tsv.
    modality: str
        Modality to select sessions. Sessions which do not include the modality will be excluded.
    restriction_path: str (path)
        Path to a tsv containing the sessions that can be included.
    variables_of_interest: List of str
        Columns that should be kept in the output tsv files.
    remove_smc: bool
        If True SMC participants are removed from the lists.
    caps_directory: str (path)
        Path to a folder of a older of a CAPS compliant dataset
    merged_tsv: str (path)
        Path to the output of clinica iotools merge-tsv if already exists
    missing_mods: str (path)
        Path to the output directory of clinica iotools check-missing-modalities if already exists
    remove_unique_session: bool
        If True, subjects with only one session are removed.
    output_dir: str (path)
        Path to the directory where the output labels.tsv will be stored.
    """

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_tsv = output_dir / "labels.tsv"

    commandline_to_json(
        {
            "bids_directory": bids_directory,
            "output_dir": output_dir,
            "diagnoses": diagnoses,
            "modality": modality,
            "restriction_path": restriction_path,
            "variables_of_interest": variables_of_interest,
            "remove_smc": remove_smc,
            "missing_mods": missing_mods,
            "merged_tsv": merged_tsv,
            "remove_unique_session": remove_unique_session_,
            "caps_directory": caps_directory,
        },
        filename="labels.json",
    )

    # Generating the output of `clinica iotools check-missing-modalities``
    missing_mods_directory = output_dir / "missing_mods"
    if missing_mods is not None:
        missing_mods_directory = missing_mods

    if not missing_mods_directory.is_dir():
        raise ValueError(
            f"The missing_mods directory doesn't exist: {missing_mods}, please give another directory."
        )

    logger.info(
        f"output of clinica iotools check-missing-modalities: {missing_mods_directory}"
    )

    # Generating the output of `clinica iotools merge-tsv `
    if not merged_tsv:
        merged_tsv = output_dir / "merged.tsv"
        if merged_tsv.is_file():
            logger.warning(
                f"A merged_tsv file already exists at {merged_tsv}. It will be used to run the command."
            )
        else:
            raise ValueError(
                "We can't find any merged tsv files, please give another path."
            )

    logger.info(f"output of clinica iotools merge-tsv: {merged_tsv}")

    # Reading files
    if not merged_tsv.is_file():
        raise ClinicaDLTSVError(f"{merged_tsv} file was not found. ")
    bids_df = pd.read_csv(merged_tsv, sep="\t", low_memory=False)
    bids_df.set_index(["participant_id", "session_id"], inplace=True)
    variables_list = []

    if "dx1" in bids_df.columns:
        bids_df.rename(columns={"dx1": "diagnosis"}, inplace=True)

    try:
        variables_list.append(find_label(bids_df.columns.values, "age"))
        variables_list.append(find_label(bids_df.columns.values, "sex"))
        variables_list.append(find_label(bids_df.columns.values, "diagnosis"))
    except ValueError:
        logger.warning(
            "The age, sex or diagnosis values were not found in the dataset."
        )

    # Cleaning NaN diagnosis
    logger.debug("Cleaning NaN diagnosis")
    bids_df = cleaning_nan_diagnoses(bids_df)

    # Checking the variables of interest
    if variables_of_interest is not None:
        variables_set = set(variables_of_interest) | set(variables_list)
        variables_list = list(variables_set)
        if not set(variables_list).issubset(set(bids_df.columns.values)):
            raise ClinicaDLArgumentError(
                f"The variables asked by the user {variables_of_interest} do not "
                f"exist in the data set."
            )

    # Loading missing modalities files
    list_files = list(missing_mods_directory.iterdir())
    missing_mods_dict = {}
    for file in list_files:
        fileext = file.suffix
        filename = file.stem
        if fileext == ".tsv":
            session = filename.split("_")[-1]
            missing_mods_df = pd.read_csv(file, sep="\t")
            if len(missing_mods_df) == 0:
                raise ClinicaDLTSVError(
                    f"Given TSV file at {file} loads an empty DataFrame."
                )

            missing_mods_df.set_index("participant_id", drop=True, inplace=True)
            missing_mods_dict[session] = missing_mods_df

    # Remove SMC patients
    if remove_smc:
        if "diagnosis_bl" in bids_df.columns.values:  # Retro-compatibility
            bids_df = bids_df[~(bids_df.diagnosis_bl == "SMC")]
        if "diagnosis_sc" in bids_df.columns.values:
            bids_df = bids_df[~(bids_df.diagnosis_sc == "SMC")]

    # Adding the field baseline_diagnosis
    bids_copy_df = copy(bids_df)
    bids_copy_df["baseline_diagnosis"] = pd.Series(
        np.zeros(len(bids_df)), index=bids_df.index
    )
    for subject, subject_df in bids_df.groupby(level=0):
        baseline_diagnosis = subject_df.loc[
            (subject, first_session(subject_df)), "diagnosis"
        ]
        bids_copy_df.loc[subject, "baseline_diagnosis"] = baseline_diagnosis

    bids_df = copy(bids_copy_df)
    variables_list.append("baseline_diagnosis")

    bids_df = bids_df[variables_list]
    if remove_unique_session_:
        bids_df = remove_unique_session(bids_df)

    variables_list.remove("baseline_diagnosis")
    output_df = bids_df[variables_list]
    output_df = infer_or_drop_diagnosis(output_df)
    output_df = diagnosis_removal(output_df, diagnoses)
    output_df = mod_selection(output_df, missing_mods_dict, modality)
    output_df = apply_restriction(output_df, restriction_path)

    output_df.reset_index()
    output_df.sort_values(by=["participant_id", "session_id"], inplace=True)
    output_df.to_csv(output_tsv, sep="\t")

    logger.info(f"Results are stored in {output_dir}.")
