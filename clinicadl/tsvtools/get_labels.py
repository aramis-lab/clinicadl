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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from clinicadl.data.datatypes.modalities import DWI, PET, Flair, Modality, T1w
from clinicadl.tsvtools.utils import (
    cleaning_nan_diagnoses,
    find_label,
    first_session,
    last_session,
    neighbour_session,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError

logger = getLogger("clinicadl.tsvtools")


def infer_or_drop_diagnosis(bids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduce the diagnosis when missing from previous and following sessions of the subject. If not identical, the session
    is dropped. Sessions with no diagnosis are also dropped when there are the last sessions of the follow-up.

    Parameters
    ----------
    bids_df: DataFrame
        Columns including ['participant_id', 'session_index', 'diagnosis'].

    Returns
    -------
    bids_copy_df: DataFrame
        Cleaned copy of the input bids_df.
    """
    bids_copy_df = copy(bids_df)
    found_diag_interpol = 0
    nb_drop = 0

    for subject, subject_df in bids_df.groupby(level=0):
        session_list = [session for _, session in subject_df.index.values]

        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), "diagnosis"]

            if isinstance(diagnosis, float):
                if session == last_session(session_list):
                    bids_copy_df.drop(index=(_, session), axis=0, inplace=True)
                    nb_drop += 1
                else:
                    prev_session = neighbour_session(session, session_list, -1)
                    prev_diagnosis = bids_df.at[(subject, prev_session), "diagnosis"]
                    while isinstance(
                        prev_diagnosis, float
                    ) and prev_session != first_session(subject_df):
                        prev_session = neighbour_session(prev_session, session_list, -1)
                        prev_diagnosis = bids_df.at[
                            (subject, prev_session), "diagnosis"
                        ]
                    post_session = neighbour_session(session, session_list, +1)
                    post_diagnosis = bids_df.at[(subject, post_session), "diagnosis"]
                    while isinstance(
                        post_diagnosis, float
                    ) and post_session != last_session(session_list):
                        post_session = neighbour_session(post_session, session_list, +1)
                        post_diagnosis = bids_df.at[
                            (subject, post_session), "diagnosis"
                        ]
                    if prev_diagnosis == post_diagnosis:
                        found_diag_interpol += 1
                        bids_copy_df.loc[
                            (subject, session), "diagnosis"
                        ] = prev_diagnosis
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
                mod_present = missing_mods_dict[
                    bids_copy_df.loc[(subject, session), "session_id"]
                ].loc[subject, mod]
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
                & (
                    restriction_df.session_id
                    == bids_copy_df.loc[(subject, session), "session_id"]
                )
            ]
            if len(subject_qc_df) != 1:
                bids_copy_df.drop((subject, session), inplace=True)
                nb_subjects += 1
    logger.info(f"Dropped subjects (apply restriction): {nb_subjects}")
    return bids_copy_df


def load_missing_mods_dict(missing_mods: Path):
    """
    Load the missing modalities files in a dictionary.

    Parameters
    ----------
    missing_mods: str (path)
        Path to the output directory of clinica iotools check-missing-modalities if already exists

    Returns
    -------
    missing_mods_dict: dictionary of str and DataFrame
        DataFrames of missing modalities
    """

    if not missing_mods.is_dir():
        raise ValueError(
            f"The missing_mods directory doesn't exist: {missing_mods}, please give another directory."
        )
    # Loading missing modalities files
    list_files = list(missing_mods.iterdir())
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
    return missing_mods_dict


def get_variables_list(
    df: pd.DataFrame, variables_of_interest: Optional[List[str]] = None
) -> list:
    variables_list = ["session_id"]

    try:
        variables_list.append(find_label(df.columns.values, "age"))
        variables_list.append(find_label(df.columns.values, "sex"))
        variables_list.append(find_label(df.columns.values, "diagnosis"))
    except ValueError:
        logger.warning(
            "The age, sex or diagnosis values were not found in the dataset."
        )

    # Checking the variables of interest
    if variables_of_interest is not None:
        variables_set = set(variables_of_interest) | set(variables_list)
        variables_list = list(variables_set)
        if not set(variables_list).issubset(set(df.columns.values)):
            raise ClinicaDLArgumentError(
                f"The variables asked by the user {variables_of_interest} do not "
                f"exist in the data set."
            )
    return variables_list


def drop_bad_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with a session_id not starting with 'ses-M'
    """

    nb_drop = 0
    for index, row in df.iterrows():
        if not row["session_id"].startswith("ses-M"):
            df.drop(index, axis=0, inplace=True)
            nb_drop += 1
    if nb_drop > 0:
        logger.warning(
            f"Dropped {nb_drop} subjects (bad session name, example ses-Nv)."
        )
    return df


def get_labels(
    bids_directory: Path,
    merged_tsv: Path,
    missing_mods: Path,
    diagnoses: List[str],
    modality: Modality = T1w(),
    variables_of_interest: Optional[List[str]] = None,
    remove_smc: bool = True,
    remove_unique_session_: bool = False,
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

    if not merged_tsv.is_file():
        raise ClinicaDLTSVError(f"{merged_tsv} file was not found. ")

    merged_df = pd.read_csv(merged_tsv, sep="\t", low_memory=False)

    # dropping rows with a session_id not starting with 'ses-M'
    merged_df = drop_bad_session(merged_df)

    # Get the session index from the session_id
    merged_df["session_index"] = (
        merged_df["session_id"].str.replace("ses-M", "").astype("int")
    )
    merged_df.set_index(["participant_id", "session_index"], inplace=True)

    # check if diagnosis column exists (written dx1 in the merged tsv file)
    if "dx1" in merged_df.columns:
        merged_df.rename(columns={"dx1": "diagnosis"}, inplace=True)

    # Getting the variables of interest
    variables_list = get_variables_list(merged_df, variables_of_interest)

    # Cleaning NaN diagnosis
    logger.debug("Cleaning NaN diagnosis")
    merged_df = cleaning_nan_diagnoses(merged_df)

    # Loading the bids merged tsv file
    missing_mods_dict = load_missing_mods_dict(missing_mods)

    # Remove SMC patients
    if remove_smc:
        if "diagnosis_bl" in merged_df.columns.values:  # Retro-compatibility
            merged_df = merged_df[~(merged_df.diagnosis_bl == "SMC")]
        if "diagnosis_sc" in merged_df.columns.values:
            merged_df = merged_df[~(merged_df.diagnosis_sc == "SMC")]

    # Adding the field baseline_diagnosis
    copy_df = copy(merged_df)
    copy_df["baseline_diagnosis"] = pd.Series(
        np.zeros(len(merged_df)), index=merged_df.index
    )
    for subject, subject_df in merged_df.groupby(level=0):
        baseline_diagnosis = subject_df.loc[
            (subject, first_session(subject_df)), "diagnosis"
        ]
        copy_df.loc[subject, "baseline_diagnosis"] = baseline_diagnosis

    merged_df = copy(copy_df)
    variables_list.append("baseline_diagnosis")

    merged_df = merged_df[variables_list]
    if remove_unique_session_:
        bids_df = remove_unique_session(bids_df)

    variables_list.remove("baseline_diagnosis")

    output_df = bids_df[variables_list]
    output_df = infer_or_drop_diagnosis(output_df)
    output_df = diagnosis_removal(output_df, diagnoses)
    output_df = mod_selection(output_df, missing_mods_dict, modality)
    # output_df = apply_restriction(output_df, restriction_path)

    output_df.reset_index(inplace=True)
    output_df.sort_values(by=["participant_id", "session_index"], inplace=True)
    output_df.drop("session_index", axis=1, inplace=True)

    return output_df
