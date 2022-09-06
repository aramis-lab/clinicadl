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
from copy import copy
from logging import getLogger
from os import path
from typing import Dict, List

import numpy as np
import pandas as pd

from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import (
    after_end_screening,
    cleaning_nan_diagnoses,
    find_label,
    first_session,
    last_session,
    neighbour_session,
)

logger = getLogger("clinicadl")


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
    nb_drop = 0

    for subject, subject_df in bids_df.groupby(level=0):
        session_list = []
        for _, session in subject_df.index.values:
            x = session[5::]
            if x.isdigit():
                session_list.append(int(x))

            else:
                subject_df.drop((subject, session), axis=0, inplace=True)
                bids_copy_df.drop((subject, session), axis=0, inplace=True)
                nb_drop += 1

        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), "diagnosis"]
            session_nb = int(session[5::])

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
                        nb_drop += 1

    logger.info(f"Inferred diagnosis: {found_diag_interpol}")
    logger.info(f"Dropped subjects (inferred diagnosis): {nb_drop}")

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
    logger.info(f"Dropped subjects (mod selection): {nb_subjects}")
    return bids_copy_df


def get_subgroup(
    bids_df: pd.DataFrame,
    horizon_time: int = 36,
    stability_dict: dict = None,
    remove_unique_session=False,
) -> pd.DataFrame:
    """
    A method to get the subgroup for each sessions depending on their stability on the time horizon

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        horizon_time: time horizon in months

    Returns:
        DataFrame with new labels
    """

    bids_df = infer_or_drop_diagnosis(bids_df)

    # Check possible double change in diagnosis in time or if ther is only one session for a subject
    # This subjects were removed in the old getlabels.
    # We can add an option to remove them, or remove them all the time or never remove them
    # We can also give two file.stv, one with unknown and unstable subjects and one without

    bids_copy_df = copy(bids_df)
    nb_subjects = 0
    nb_unique = 0

    # Do not take into account the case of missing diag = nan

    for subject, subject_df in bids_df.groupby(level=0):

        session_list = [int(session[5:]) for _, session in subject_df.index.values]

        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), "diagnosis"]
            # print(diagnosis)
            # print(stability_dict)
            diagnosis_dict = stability_dict[diagnosis]
            session_nb = int(session[5::])
            horizon_session_nb = session_nb + horizon_time
            horizon_session = "ses-M" + str(horizon_session_nb)
            # print(session, '-->', horizon_session)

            # CASE 1 : if the  session after 'horizon_time' months is a session the subject has done
            if horizon_session_nb in session_list:
                horizon_diagnosis = subject_df.loc[
                    (subject, horizon_session), "diagnosis"
                ]
                horizon_diagnosis_dict = stability_dict[horizon_diagnosis]

                if horizon_diagnosis_dict > diagnosis_dict:
                    update_diagnosis = "p"
                elif horizon_diagnosis_dict < diagnosis_dict:
                    update_diagnosis = "r"
                elif horizon_diagnosis_dict == diagnosis_dict:
                    update_diagnosis = "s"

            # CASE 2 : if the session after 'horizon_time' months doesn't exist because it is after the last session of the subject
            elif after_end_screening(horizon_session_nb, session_list):
                last_diagnosis = subject_df.loc[
                    (subject, last_session(session_list)), "diagnosis"
                ]
                # This section must be discussed --> removed in Jorge's paper
                last_diagnosis_dict = stability_dict[last_diagnosis]
                if last_diagnosis_dict > diagnosis_dict:
                    update_diagnosis = "p"
                elif last_diagnosis_dict < diagnosis_dict:
                    update_diagnosis = "r"
                elif last_diagnosis_dict == diagnosis_dict:
                    update_diagnosis = "s"

            # CASE 3 : if the session after 'horizon_time' months doesn't exist but ther are sessions before and after this time.
            else:
                prev_session = neighbour_session(horizon_session_nb, session_list, -1)
                post_session = neighbour_session(horizon_session_nb, session_list, +1)

                prev_diagnosis = subject_df.loc[(subject, prev_session), "diagnosis"]
                prev_diagnosis_dict = stability_dict[prev_diagnosis]

                if prev_diagnosis_dict > diagnosis_dict:
                    update_diagnosis = "p"
                elif prev_diagnosis_dict < diagnosis_dict:
                    update_diagnosis = "r"
                elif prev_diagnosis_dict == diagnosis_dict:
                    post_diagnosis = subject_df.loc[
                        (subject, post_session), "diagnosis"
                    ]
                    post_diagnosis_dict = stability_dict[post_diagnosis]
                    if post_diagnosis_dict > diagnosis_dict:
                        update_diagnosis = "p"
                    elif post_diagnosis_dict < diagnosis_dict:
                        update_diagnosis = "r"
                    elif post_diagnosis_dict == diagnosis_dict:
                        update_diagnosis = "s"
            bids_copy_df.loc[(subject, session), "group"] = diagnosis
            bids_copy_df.loc[(subject, session), "subgroup"] = (
                update_diagnosis + diagnosis
            )
        # Remove subject with a unique session if wanted
        if remove_unique_session is True:

            nb_session = len(session_list)
            if nb_session == 1:
                bids_copy_df.drop((subject, session_list[0]), inplace=True)
                subject_df.drop((subject, session_list[0]), inplace=True)
                nb_unique += 1

        # Add unknown subgroup for each last_session
        session_list = [int(session[5::]) for _, session in subject_df.index.values]
        last_session_str = last_session(session_list)
        diagnosis = bids_copy_df.loc[(subject, last_session_str), "diagnosis"]
        bids_copy_df.loc[(subject, last_session_str), "subgroup"] = "uk" + diagnosis

        # Add unstable session for subjects with multiple regression or conversion
        # The subjects will be unstable only from the time of the conversion (if regression before) or regression (if conversion before)
        session_list.sort()
        status = 0
        unstable = False
        for session in session_list:
            if session < 10:
                ses_str = "ses-M0"
            else:
                ses_str = "ses-M"
            session_str = ses_str + str(session)
            subgroup_str = bids_copy_df.loc[(subject, session_str), "subgroup"]
            subgroup_str = subgroup_str[:1]
            if subgroup_str == "p":
                if status < 0:
                    diagnosis = bids_copy_df.loc[(subject, session_str), "group"]
                    bids_copy_df.loc[(subject, session_str), "subgroup"] = (
                        "us" + diagnosis
                    )
                    unstable = True
                else:
                    status = 1
            if subgroup_str == "r":
                if status > 0:
                    diagnosis = bids_copy_df.loc[(subject, session_str), "group"]
                    bids_copy_df.loc[(subject, session_str), "subgroup"] = (
                        "us" + diagnosis
                    )
                    unstable = True
                else:
                    status = -1
        if unstable:
            nb_subjects += 1
            for session in session_list:
                if session < 10:
                    ses_str = "ses-M0"
                else:
                    ses_str = "ses-M"
                session_str = ses_str + str(session)
                subgroup_str = bids_copy_df.loc[(subject, session_str), "subgroup"]
                subgroup_str = subgroup_str[:1]
                if subgroup_str == "s":
                    diagnosis = bids_copy_df.loc[(subject, session_str), "group"]
                    bids_copy_df.loc[(subject, session_str), "subgroup"] = (
                        "us" + diagnosis
                    )

    logger.info(f"Dropped subjects (unique session): {nb_unique}")
    logger.info(f"Unstable subjects: {nb_subjects}")

    return bids_copy_df


def diagnosis_removal(bids_df: pd.DataFrame, diagnosis_list: List[str]) -> pd.DataFrame:
    """
    Removes sessions for which the diagnosis is not in the list provided (avoid to keep rMCI and pMCI in sMCI lists).

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'group', 'subgroup']
        diagnosis_list: list of diagnoses that will be removed

    Returns:
        cleaned DataFrame
    """

    output_df = copy(bids_df)
    nb_subjects = 0
    for subject, subject_df in bids_df.groupby(level=0):
        for (_, session) in subject_df.index.values:
            if subject_df.loc[(subject, session), "group"] not in diagnosis_list:
                if subject_df.loc[(subject, session), "subgroup"] not in diagnosis_list:
                    output_df.drop((subject, session), inplace=True)
                    nb_subjects += 1

    logger.info(f"Dropped subjects (diagnoses): {nb_subjects}")
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
    bids_directory: str,
    results_directory: str,
    diagnoses: List[str],
    modality: str = "t1w",
    restriction_path: str = None,
    time_horizon: int = 36,
    variables_of_interest: List[str] = None,
    remove_smc: bool = True,
    caps_directory: str = None,
    merge_tsv: str = None,
    missing_mods: str = None,
    remove_unique_session: bool = False,
):
    """
    Writes one TSV file based on merged_tsv and missing_mods.
    Calculates the subgroup of each session.


    Args:
        bids_directory: Path to the folder containing the dataset in a BIDS hierarchy.
        results_directory: Path to the folder where merge-tsv, missing-mod and getlabels files will be saved.
        diagnoses: Labels that must be extracted from merged_tsv.
        stability_dict: List of all the diagnoses that can be encountered in order of the disease progression.
        modality: Modality to select sessions. Sessions which do not include the modality will be excluded.
        restriction_path: Path to a tsv containing the sessions that can be included.
        time_horizon: Time horizon to analyse stability of MCI subjects.
        variables_of_interest: columns that should be kept in the output tsv files.
        remove_smc: if True SMC participants are removed from the lists.
        caps_directory: Path to a folder of a older of a CAPS compliant dataset
    """

    commandline_to_json(
        {
            "bids_directory": bids_directory,
            "output_dir": results_directory,
            "diagnoses": diagnoses,
            "modality": modality,
            "restriction_path": restriction_path,
            "time_horizon": time_horizon,
            "variables_of_interest": variables_of_interest,
            "remove_smc": remove_smc,
            "caps": caps_directory,
            "missing_mods": missing_mods,
            "merge_tsv": merge_tsv,
            "remove_unique_session": remove_unique_session,
        },
        filename="getlabels.json",
    )

    import os

    from clinica.iotools.utils.data_handling import (
        compute_missing_mods,
        create_merge_file,
    )
    from clinica.utils.inputs import check_bids_folder

    # Create the results directory
    os.makedirs(results_directory, exist_ok=True)

    # Generating the output of `clinica iotools check-missing-modalities``
    missing_mods_path = os.path.join(results_directory, "missing_mods")
    if missing_mods is not None:
        missing_mods_path = missing_mods
    # print(bids_directory)
    if not os.path.exists(missing_mods_path):
        # print(os.path.join(results_directory, "missing_mods"))
        logger.info("create missing modalities directories")
        check_bids_folder(bids_directory)
        compute_missing_mods(bids_directory, missing_mods_path, "missing_mods")
    # print(results_directory)

    # Generating the output of `clinica iotools merge-tsv `
    merge_tsv_path = os.path.join(results_directory, "merge.tsv")
    if merge_tsv is not None:
        merge_tsv_path = merge_tsv
    elif not os.path.exists(merge_tsv_path):
        logger.info("create merge tsv")
        check_bids_folder(bids_directory)
        create_merge_file(
            bids_directory,
            results_directory + "/merge.tsv",
            caps_dir=caps_directory,
            pipelines=None,
            ignore_scan_files=None,
            ignore_sessions_files=None,
            volume_atlas_selection=None,
            freesurfer_atlas_selection=None,
            pvc_restriction=None,
            tsv_file=None,
            group_selection=False,
            tracers_selection=False,
        )

    # Reading files
    bids_df = pd.read_csv(merge_tsv_path, sep="\t")
    bids_df.set_index(["participant_id", "session_id"], inplace=True)
    variables_list = []
    try:
        variables_list.append(find_label(bids_df.columns.values, "age"))
        variables_list.append(find_label(bids_df.columns.values, "sex"))
        variables_list.append(find_label(bids_df.columns.values, "diagnosis"))
    except ValueError:
        logger.warning(
            "The age, sex or diagnosis values were not found in the dataset."
        )
    # Cleaning NaN diagnosis
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
    missing_mods_directory = missing_mods_path
    list_files = os.listdir(missing_mods_directory)
    missing_mods_dict = {}

    for file in list_files:
        filename, fileext = path.splitext(file)
        if fileext == ".tsv":
            session = filename.split("_")[-1]
            missing_mods_df = pd.read_csv(
                path.join(missing_mods_directory, file), sep="\t"
            )
            if len(missing_mods_df) == 0:
                raise ClinicaDLTSVError(
                    f"Given TSV file at {path.join(missing_mods_directory, file)} loads an empty DataFrame."
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
        baseline_diagnosis = subject_df.loc[(subject, "ses-M00"), "diagnosis"]
        bids_copy_df.loc[subject, "baseline_diagnosis"] = baseline_diagnosis

    bids_df = copy(bids_copy_df)
    bids_df["group"] = "UK"
    bids_df["subgroup"] = "UK"

    variables_list.append("group")
    variables_list.append("subgroup")
    variables_list.append("baseline_diagnosis")

    bids_df = bids_df[variables_list]

    stability_dict = {"CN": 0, "MCI": 1, "AD": 2}
    output_df = get_subgroup(bids_df, time_horizon, stability_dict=stability_dict)

    variables_list.remove("baseline_diagnosis")
    variables_list.remove("diagnosis")

    output_df = output_df[variables_list]
    output_df = diagnosis_removal(output_df, diagnoses)
    output_df = mod_selection(output_df, missing_mods_dict, modality)
    output_df = apply_restriction(output_df, restriction_path)

    output_df.to_csv(path.join(results_directory, "getlabels.tsv"), sep="\t")
