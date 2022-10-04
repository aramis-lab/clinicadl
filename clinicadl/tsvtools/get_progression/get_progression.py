import os
from copy import copy
from logging import getLogger
from os import path
from typing import Dict, List

import numpy as np
import pandas as pd

from clinicadl.tsvtools.get_labels import infer_or_drop_diagnosis
from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import (
    after_end_screening,
    cleaning_nan_diagnoses,
    find_label,
    first_session,
    last_session,
    merge_tsv_reader,
    neighbour_session,
)

logger = getLogger("clinicadl")


def get_progression(
    formatted_data_tsv: str,
    horizon_time: int = 36,
    stability_dict: dict = None,
):

    """
    A method to get the subgroup for each sessions depending on their stability on the time horizon

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id']
        horizon_time: time horizon in months

    Returns:
        DataFrame with new labels
    """

    # Reading files
    bids_df = merge_tsv_reader(formatted_data_tsv)

    if "diagnosis" not in bids_df.columns:
        data_directory = Path(output_tsv).parents[0]
        formatted_data_tsv = results_directory / "labels.tsv"

        metadata_df = merge_tsv_reader(formatted_data_tsv)
        bids_df.rename(columns={"dx1": "diagnosis"}, inplace=True)
    bids_df.set_index(["participant_id", "session_id"], inplace=True)

    bids_df = infer_or_drop_diagnosis(bids_df)

    # Check possible double change in diagnosis in time or if ther is only one session for a subject
    # This subjects were removed in the old getlabels.
    # We can add an option to remove them, or remove them all the time or never remove them
    # We can also give two file.stv, one with unknown and unstable subjects and one without

    stability_dict = {"CN": 0, "MCI": 1, "AD": 2, "Dementia": 2}
    nb_subjects = 0
    # print(bids_df.columns.values)
    # if "group" is in bids_df.columns.values :
    #     diagnosis_str = "group"
    # elif "diagnosis" is in bids_df.columns.values :
    #     diagnosis_str = "diagnosis"

    # #bids_df["group"] = "UK"
    bids_df["progression"] = "UK"
    bids_copy_df = copy(bids_df)

    # Do not take into account the case of missing diag = nan

    for subject, subject_df in bids_df.groupby(level=0):
        session_list = [session for _, session in subject_df.index.values]
        session_list.sort()
        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), "diagnosis"]
            diagnosis_dict = stability_dict[diagnosis]
            session_nb = int(session[5::])
            horizon_session_nb = session_nb + horizon_time
            if horizon_session_nb < 10:
                horizon_session = "ses-M00" + str(horizon_session_nb)
            elif 10 <= horizon_session_nb < 100:
                horizon_session = "ses-M0" + str(horizon_session_nb)
            else:
                horizon_session = "ses-M" + str(horizon_session_nb)
            # print(session, '-->', horizon_session)exit

            # CASE 1 : if the  session after 'horizon_time' months is a session the subject has done
            if horizon_session in session_list:
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
            elif after_end_screening(horizon_session, session_list):
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
                prev_session = neighbour_session(horizon_session, session_list, -1)
                post_session = neighbour_session(horizon_session, session_list, +1)

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
            bids_copy_df.loc[(subject, session), "progression"] = update_diagnosis

        # Add unstable session for subjects with multiple regression or conversion
        # The subjects will be unstable only from the time of the conversion (if regression before) or regression (if conversion before)

        status = 0
        unstable = False
        for session_str in session_list:
            subgroup_str = bids_copy_df.loc[(subject, session_str), "progression"]
            if subgroup_str == "p":
                if status < 0:
                    unstable = True
                status = 1
            if subgroup_str == "r":
                if status > 0:
                    unstable = True
                status = -1
        if unstable:
            nb_subjects += 1
            for session_str in session_list:
                bids_copy_df.loc[(subject, session_str), "progression"] = "us"

        # Add unknown subgroup for each last_session
        session_list = [session for _, session in subject_df.index.values]

        last_session_str = last_session(session_list)
        if bids_copy_df.loc[(subject, last_session_str), "progression"] != "us":
            bids_copy_df.loc[(subject, last_session_str), "progression"] = "uk"

    logger.info(f"Unstable subjects: {nb_subjects}")

    bids_copy_df.to_csv(formatted_data_tsv, sep="\t")
