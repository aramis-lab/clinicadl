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
import os
from cmath import nan
from copy import copy
from logging import getLogger
from os import path
from typing import Dict, List

import numpy as np
import pandas as pd
from requests import get

from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import (
    after_end_screening,
    find_label,
    first_session,
    last_session,
    neighbour_session,
)

logger = getLogger("clinicadl")


def cleaning_nan_diagnoses(bids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Printing the number of missing diagnoses and filling it partially for ADNI datasets

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']

    Returns:
        cleaned DataFrame
    """
    bids_copy_df = copy(bids_df)

    # Look for the diagnosis in another column in ADNI
    if "adni_diagnosis_change" in bids_df.columns:
        change_dict = {
            1: "CN",
            2: "MCI",
            3: "AD",
            4: "MCI",
            5: "AD",
            6: "AD",
            7: "CN",
            8: "MCI",
            9: "CN",
            -1: np.nan,
        }

        missing_diag = 0
        found_diag = 0

        for subject, session in bids_df.index.values:
            diagnosis = bids_df.loc[(subject, session), "diagnosis"]
            if isinstance(diagnosis, float):
                missing_diag += 1
                change = bids_df.loc[(subject, session), "adni_diagnosis_change"]
                if not np.isnan(change) and change != -1:
                    found_diag += 1
                    bids_copy_df.loc[(subject, session), "diagnosis"] = change_dict[
                        change
                    ]

    else:
        missing_diag = 0
        found_diag = 0

        for subject, session in bids_df.index.values:
            diagnosis = bids_df.loc[(subject, session), "diagnosis"]
            if isinstance(diagnosis, float):
                missing_diag += 1

    logger.debug(f"Missing diagnoses: {missing_diag}")
    logger.debug(f"Missing diagnoses not found: {missing_diag - found_diag}")

    return bids_copy_df


def infer_or_drop_diagnosis(bids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduce the diagnosis when missing from previous and following sessions of the subject. If not identical, the session
    is dropped. Sessions with no diagnosis are also dropped when there are the last sessions of the follow-up.

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']

    Returns:
        cleaned DataFrame
    """
    bids_copy_df = copy(bids_df)
    found_diag_interpol = 0

    for subject, subject_df in bids_df.groupby(level=0):
        session_list = []
        for _, session in subject_df.index.values:
            x = session[5::]
            if x.isdigit():
                session_list.append(int(x))

            else:
                subject_df.drop((subject, session), axis=0, inplace=True)
                bids_copy_df.drop((subject, session), axis=0, inplace=True)

        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), "diagnosis"]
            session_nb = int(session[5::])

            if isinstance(diagnosis, float):
                if session == last_session(session_list):
                    bids_copy_df.drop(index=(_, session), axis=0, inplace=True)
                else:
                    prev_session = neighbour_session(session_nb, session_list, -1)
                    prev_diagnosis = bids_df.loc[(subject, prev_session), "diagnosis"]
                    while isinstance(
                        prev_diagnosis, float
                    ) and prev_session != first_session(subject_df):
                        prev_session = neighbour_session(
                            int(prev_session[5::]), session_list, -1
                        )
                        prev_diagnosis = bids_df.loc[
                            (subject, prev_session), "diagnosis"
                        ]
                    post_session = neighbour_session(session_nb, session_list, +1)
                    post_diagnosis = bids_df.loc[(subject, post_session), "diagnosis"]
                    while isinstance(
                        post_diagnosis, float
                    ) and post_session != last_session(session_list):
                        post_session = neighbour_session(
                            int(post_session[5::]), session_list, +1
                        )
                        post_diagnosis = bids_df.loc[
                            (subject, post_session), "diagnosis"
                        ]
                    if prev_diagnosis == post_diagnosis:
                        found_diag_interpol += 1
                        bids_copy_df.loc[
                            (subject, session), "diagnosis"
                        ] = prev_diagnosis
                    else:
                        bids_copy_df.drop((subject, session), inplace=True)

    logger.info(f"Inferred diagnosis: {found_diag_interpol}")

    return bids_copy_df


def mod_selection(
    bids_df: pd.DataFrame, missing_mods_dict: Dict[str, pd.DataFrame], mod: str = "t1w"
) -> pd.DataFrame:
    """
    Select only sessions for which the modality is present

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        missing_mods_dict: dictionary of the DataFrames of missing modalities
        mod: the modality used for selection

    Returns:
        DataFrame
    """
    bids_copy_df = copy(bids_df)
    if mod is not None:
        for subject, session in bids_df.index.values:
            try:
                mod_present = missing_mods_dict[session].loc[subject, mod]
                if not mod_present:
                    bids_copy_df.drop((subject, session), inplace=True)
            except KeyError:
                bids_copy_df.drop((subject, session), inplace=True)

    return bids_copy_df


def get_subgroup(
    bids_df: pd.DataFrame, horizon_time: int = 36, stability_dict_test: dict = None
) -> pd.DataFrame:
    """
    A method to label all MCI sessions depending on their stability on the time horizon

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        horizon_time: time horizon in months

    Returns:
        DataFrame with new labels
    """

    bids_df = cleaning_nan_diagnoses(bids_df)
    bids_df = infer_or_drop_diagnosis(bids_df)

    # Check possible double change in diagnosis in time or if ther is only one session for a subject
    # This subjects were removed in the old getlabels.
    # We can add an option to remove them, or remove them all the time or never remove them
    # We can also give two file.stv, one with unknown and unstable subjects and one without

    bids_copy_df = copy(bids_df)
    nb_subjects = 0
    for subject, subject_df in bids_df.groupby(level=0):
        session_list = [int(session[5::]) for _, session in subject_df.index.values]
        if len(session_list) == 1:
            diagnosis = bids_copy_df.loc[(subject, "ses-M00"), "diagnosis"]
            bids_copy_df.loc[(subject, "ses-M00"), "subgroup"] = "uk" + diagnosis
            bids_copy_df.loc[(subject, "ses-M00"), "group"] = diagnosis
            # bids_copy_df.drop(subject, inplace=True)
            nb_subjects += 1
        else:
            session_list.sort()
            diagnosis_list = []

            for session in session_list:
                if session < 10:
                    ses_str = "ses-M0"
                else:
                    ses_str = "ses-M"
                diagnosis_list.append(
                    bids_df.loc[(subject, ses_str + str(session)), "diagnosis"]
                )
            bl_diagnosis = diagnosis_list[0]
            bl_diagnosis_dict = stability_dict_test[bl_diagnosis]
            status = 0
            for diagnosis in diagnosis_list:
                diagnosis_dict = stability_dict_test[diagnosis]
                if diagnosis_dict > bl_diagnosis_dict:
                    if status < 0:
                        bids_copy_df.loc[subject, "subgroup"] = "us" + diagnosis
                        for _, session in subject_df.index.values:
                            bids_copy_df.loc[
                                (subject, session), "group"
                            ] = bids_copy_df.loc[(subject, session), "diagnosis"]
                        # bids_copy_df.drop(subject, inplace=True)
                        nb_subjects += 1
                    bl_diagnosis_dict = diagnosis_dict
                    status = 1

                elif diagnosis_dict < bl_diagnosis_dict:
                    if status > 0:
                        bids_copy_df.loc[subject, "subgroup"] = "us" + diagnosis
                        for _, session in subject_df.index.values:
                            bids_copy_df.loc[
                                (subject, session), "group"
                            ] = bids_copy_df.loc[(subject, session), "diagnosis"]
                        # bids_copy_df.drop(subject, inplace=True)
                        nb_subjects += 1
                    bl_diagnosis_dict = diagnosis_dict
                    status = -1

    logger.info(f"Dropped subjects: {nb_subjects}")

    # Do not take into account the case of missing diag = nan

    for subject, subject_df in bids_df.groupby(level=0):

        session_list = [int(session[5:]) for _, session in subject_df.index.values]

        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), "diagnosis"]
            subgroup = subject_df.loc[(subject, session), "subgroup"]
            # Check if the diagnosis is unstable or unknown
            if subgroup[:2] != "us" and subgroup[:2] != "uk":
                diagnosis_dict = stability_dict_test[diagnosis]
                session_nb = int(session[5::])
                horizon_session_nb = session_nb + horizon_time
                horizon_session = "ses-M" + str(horizon_session_nb)
                # print(session, '-->', horizon_session)

                # CASE 1 : if the  session after 'horizon_time' months is a session the subject has done
                if horizon_session_nb in session_list:
                    horizon_diagnosis = subject_df.loc[
                        (subject, horizon_session), "diagnosis"
                    ]
                    horizon_diagnosis_dict = stability_dict_test[horizon_diagnosis]

                    if horizon_diagnosis_dict > diagnosis_dict:
                        update_diagnosis = "p"
                    elif horizon_diagnosis_dict < diagnosis_dict:
                        update_diagnosis = "r"
                    elif horizon_diagnosis_dict == diagnosis_dict:
                        update_diagnosis = "s"

                # CASE 2 : if the session after 'horizon_time' months doesn't exist because it is after the last session of the subject
                elif after_end_screening(horizon_session_nb, session_list):
                    # Two situations, change in last session AD or CN --> pMCI or rMCI
                    # Last session MCI --> uMCI
                    last_diagnosis = subject_df.loc[
                        (subject, last_session(session_list)), "diagnosis"
                    ]
                    # This section must be discussed --> removed in Jorge's paper
                    last_diagnosis_dict = stability_dict_test[last_diagnosis]
                    if last_diagnosis_dict > diagnosis_dict:
                        update_diagnosis = "p"
                    elif last_diagnosis_dict < diagnosis_dict:
                        update_diagnosis = "r"
                    elif last_diagnosis_dict == diagnosis_dict:
                        update_diagnosis = "s"

                # CASE 3 : if the session after 'horizon_time' months doesn't exist but ther are sessions before and after this time.
                else:
                    prev_session = neighbour_session(
                        horizon_session_nb, session_list, -1
                    )
                    post_session = neighbour_session(
                        horizon_session_nb, session_list, +1
                    )

                    prev_diagnosis = subject_df.loc[
                        (subject, prev_session), "diagnosis"
                    ]
                    prev_diagnosis_dict = stability_dict_test[prev_diagnosis]

                    if prev_diagnosis_dict > diagnosis_dict:
                        update_diagnosis = "p"
                    elif prev_diagnosis_dict < diagnosis_dict:
                        update_diagnosis = "r"
                    elif prev_diagnosis_dict == diagnosis_dict:
                        post_diagnosis = subject_df.loc[
                            (subject, post_session), "diagnosis"
                        ]
                        post_diagnosis_dict = stability_dict_test[post_diagnosis]
                        if post_diagnosis_dict > diagnosis_dict:
                            update_diagnosis = "p"
                        elif post_diagnosis_dict < diagnosis_dict:
                            update_diagnosis = "r"
                        elif post_diagnosis_dict == diagnosis_dict:
                            update_diagnosis = "s"

                subgroup = bids_copy_df.loc[(subject, session), "subgroup"] == "UK"
                if update_diagnosis == "r" or update_diagnosis == "p" or subgroup:
                    bids_copy_df.loc[(subject), "subgroup"] = (
                        update_diagnosis + diagnosis
                    )
                bids_copy_df.loc[(subject, session), "group"] = diagnosis

    return bids_copy_df


def diagnosis_removal(bids_df: pd.DataFrame, diagnosis_list: List[str]) -> pd.DataFrame:
    """
    Removes subjects whom last diagnosis is in the list provided (avoid to keep rMCI and pMCI in sMCI lists).

    Args:
        MCI_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        diagnosis_list: list of diagnoses that will be removed

    Returns:
        cleaned DataFrame
    """

    output_df = copy(bids_df)

    # Remove subjects who regress to CN label, even late in the follow-up
    for subject, subject_df in bids_df.groupby(level=0):
        drop = True
        if subject_df.loc[(subject, "ses-M00"), "subgroup"] not in diagnosis_list:
            for (_, session) in subject_df.index.values:
                if subject_df.loc[(subject, session), "group"] in diagnosis_list:
                    drop = False
            if drop:
                output_df.drop(subject, inplace=True)

    return output_df


def apply_restriction(bids_df: pd.DataFrame, restriction_path: str) -> pd.DataFrame:
    """
    Application of a restriction (for example after the removal of some subjects after a preprocessing pipeline)

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        restriction_path: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis'] including
            all the sessions that can be included

    Returns:
        The restricted DataFrame
    """
    bids_copy_df = copy(bids_df)

    if restriction_path is not None:
        restriction_df = pd.read_csv(restriction_path, sep="\t")

        for subject, session in bids_df.index.values:
            subject_qc_df = restriction_df[
                (restriction_df.participant_id == subject)
                & (restriction_df.session_id == session)
            ]
            if len(subject_qc_df) != 1:
                bids_copy_df.drop((subject, session), inplace=True)

    return bids_copy_df


def get_labels(
    merged_tsv: str,
    missing_mods: str,
    results_path: str,
    diagnoses: List[str],
    modality: str = "t1w",
    restriction_path: str = None,
    time_horizon: int = 36,
    variables_of_interest: List[str] = None,
    remove_smc: bool = True,
):
    """
    Writes one TSV file per label in diagnoses argument based on merged_tsv and missing_mods.

    Args:
        merged_tsv: Path to the file obtained by the command clinica iotools merge-tsv.
        missing_mods: Path to the folder where the outputs of clinica iotools check-missing-modalities are.
        results_path: Path to the folder where tsv files are extracted.
        diagnoses: Labels that must be extracted from merged_tsv.
        modality: Modality to select sessions. Sessions which do not include the modality will be excluded.
        restriction_path: Path to a tsv containing the sessions that can be included.
        time_horizon: Time horizon to analyse stability of MCI subjects.
        variables_of_interest: columns that should be kept in the output tsv files.
        remove_smc: if True SMC participants are removed from the lists.
    """

    commandline_to_json(
        {
            "output_dir": results_path,
            "merged_tsv": merged_tsv,
            "missing_mods": missing_mods,
            "diagnoses": diagnoses,
            "modality": modality,
            "restriction_path": restriction_path,
            "time_horizon": time_horizon,
            "variables_of_interest": variables_of_interest,
            "remove_smc": remove_smc,
        },
        filename="getlabels.json",
    )

    stability_dict_test = {"CN": 0, "MCI": 1, "Dementia": 2}

    # Reading files
    bids_df = pd.read_csv(merged_tsv, sep="\t")
    bids_df.set_index(["participant_id", "session_id"], inplace=True)
    variables_list = ["diagnosis"]
    try:
        variables_list.append(find_label(bids_df.columns.values, "age"))
        variables_list.append(find_label(bids_df.columns.values, "sex"))
    except ValueError:
        logger.warning("The age or sex values were not found in the dataset.")
    if variables_of_interest is not None:
        variables_set = set(variables_of_interest) | set(variables_list)
        variables_list = list(variables_set)
        if not set(variables_list).issubset(set(bids_df.columns.values)):
            raise ClinicaDLArgumentError(
                f"The variables asked by the user {variables_of_interest} do not "
                f"exist in the data set."
            )

    # Loading missing modalities files
    list_files = os.listdir(missing_mods)
    missing_mods_dict = {}

    for file in list_files:
        filename, fileext = path.splitext(file)
        if fileext == ".tsv":
            session = filename.split("_")[-1]
            missing_mods_df = pd.read_csv(path.join(missing_mods, file), sep="\t")
            if len(missing_mods_df) == 0:
                raise ClinicaDLTSVError(
                    f"Given TSV file at {path.join(missing_mods, file)} loads an empty DataFrame."
                )

            missing_mods_df.set_index("participant_id", drop=True, inplace=True)
            missing_mods_dict[session] = missing_mods_df

    # Creating results path
    os.makedirs(results_path, exist_ok=True)

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
        baseline_diagnosis = subject_df.loc[(subject, "ses-M00"), "diagnosis"]
        bids_copy_df.loc[subject, "baseline_diagnosis"] = baseline_diagnosis

    bids_df = copy(bids_copy_df)

    bids_df["group"] = "UK"
    bids_df["subgroup"] = "UK"
    variables_list.append("group")
    variables_list.append("subgroup")
    variables_list.append("baseline_diagnosis")
    bids_df = bids_df[variables_list]

    logger.info("Beginning of the selection")
    output_df = get_subgroup(
        bids_df, time_horizon, stability_dict_test=stability_dict_test
    )
    variables_list.remove("baseline_diagnosis")
    variables_list.remove("diagnosis")
    output_df = output_df[variables_list]
    output_df = diagnosis_removal(output_df, diagnoses)
    output_df = mod_selection(output_df, missing_mods_dict, modality)
    output_df = apply_restriction(output_df, restriction_path)

    output_df.to_csv(path.join(results_path, "getlabels.tsv"), sep="\t")
