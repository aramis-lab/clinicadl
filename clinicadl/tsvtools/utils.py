# coding: utf8

from copy import copy
from logging import getLogger
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from clinicadl.dictionary.words import (
    DATASET_ID,
    FIRST_INDEX,
    LAST_INDEX,
    N_SAMPLES,
    PARTICIPANT_ID,
    SESSION_ID,
)
from clinicadl.utils.exceptions import ClinicaDLTSVError

logger = getLogger("clinicadl")


def remove_non_empty_dir(dir_path: Path):
    """
    Remove a non-empty directory using only pathlib.

    Parameters
    ----------
    dir_path : Path
        Path to the directory to remove.
    """
    if dir_path.exists() and dir_path.is_dir():
        for item in dir_path.iterdir():  # Iterate through directory contents
            if item.is_dir():
                remove_non_empty_dir(item)  # Recursively remove subdirectories
            else:
                item.unlink()  # Remove files
        dir_path.rmdir()  # Remove the now-empty directory
    else:
        print(f"{dir_path} does not exist or is not a directory.")


def merged_tsv_reader(merged_tsv_path: Path) -> pd.DataFrame:
    if not merged_tsv_path.is_file():
        raise ClinicaDLTSVError(f"{merged_tsv_path} file was not found. ")

    bids_df = pd.read_csv(merged_tsv_path, sep="\t")

    # To handle bids with 2 and 3 digits
    bids_df["session_id"] = bids_df["session_id"].apply(
        lambda session: (
            session[:5] + "0" + session[5:7] if len(session) == 7 else session
        )
    )

    return bids_df


def neighbour_session(session, session_list, neighbour):
    if session not in session_list:
        temp_list = session_list + [session]
        temp_list.sort()
    else:
        temp_list = copy(session_list)
        temp_list.sort()
    index_session = temp_list.index(session)

    if index_session + neighbour < 0 or index_session + neighbour >= len(temp_list):
        return None
    else:
        return temp_list[index_session + neighbour]


def after_end_screening(session, session_list):
    if session in session_list:
        return False
    else:
        temp_list = session_list + [session]
        temp_list.sort()
        index_session = temp_list.index(session)
        return index_session == len(temp_list) - 1


def last_session(session_list):
    temp_list = copy(session_list)
    temp_list.sort()
    return temp_list[-1]


def complementary_list(total_list, sub_list):
    result_list = []
    for element in total_list:
        if element not in sub_list:
            result_list.append(element)
    return result_list


def first_session(subject_df):
    session_list = [session for _, session in subject_df.index.values]
    session_list.sort()
    return session_list[0]


def next_session(subject_df, session_orig):
    session_list = [session for _, session in subject_df.index.values]
    session_list.sort()

    index = session_list.index(session_orig)
    if index < len(session_list) - 1:
        return session_list[index + 1]
    else:
        raise IndexError("The argument session is the last session")


def add_demographics(df, demographics_df, diagnosis) -> pd.DataFrame:
    out_df = pd.DataFrame()
    tmp_demo_df = copy(demographics_df)
    tmp_demo_df.reset_index(inplace=True)
    for idx in df.index.values:
        participant = df.loc[idx, "participant_id"]
        session = df.loc[idx, "session_id"]
        row_df = tmp_demo_df[
            (tmp_demo_df.participant_id == participant)
            & (tmp_demo_df.session_id == session)
        ]
        out_df = pd.concat([out_df, row_df])
    out_df.reset_index(inplace=True, drop=True)
    out_df.diagnosis = [diagnosis] * len(out_df)
    return out_df


def remove_unicity(values_list):
    """Count the values of each class and label all the classes with only one label under the same label."""
    unique_classes, counts = np.unique(values_list, return_counts=True)
    one_sub_classes = unique_classes[(counts == 1)]
    for class_element in one_sub_classes:
        values_list[values_list.index(class_element)] = unique_classes.min()

    return values_list


def category_conversion(values_list) -> List[int]:
    values_np = np.array(values_list)
    unique_classes = np.unique(values_np)
    for index, unique_class in enumerate(unique_classes):
        values_np[values_np == unique_class] = index + 1

    return values_np.astype(int).tolist()


def find_label(labels_list, target_label):
    if target_label in labels_list:
        return target_label
    else:
        min_length = np.inf
        found_label = None
        for label in labels_list:
            if target_label.lower() in label.lower() and min_length > len(label):
                min_length = len(label)
                found_label = label
        if found_label is None:
            raise ClinicaDLTSVError(
                f"No label was found in {labels_list} for target label {target_label}."
            )

        return found_label


def remove_sub_labels(
    diagnosis_df, sub_labels, diagnosis_df_paths: list, results_path: Path
):
    supplementary_diagnoses = []

    logger.debug("Before subjects removal")
    sub_df = (
        diagnosis_df.reset_index().groupby("participant_id")["session_id"].nunique()
    )
    logger.debug(f"{len(sub_df)} subjects, {len(diagnosis_df)} scans")

    for label in sub_labels:
        if Path(f"{label}.tsv") in diagnosis_df_paths:
            sub_diag_df = pd.read_csv(results_path / f"{label}.tsv", sep="\t")
            sub_diag_baseline_df = extract_baseline(sub_diag_df, label)
            for idx in sub_diag_baseline_df.index.values:
                subject = sub_diag_baseline_df.loc[idx, "participant_id"]
                diagnosis_df.drop(subject, inplace=True, level=0)
            supplementary_diagnoses.append(label)

            logger.debug(
                f"Removed {len(sub_diag_baseline_df)} subjects based on {label} label"
            )
            sub_df = (
                diagnosis_df.reset_index()
                .groupby("participant_id")["session_id"]
                .nunique()
            )
            logger.debug(f"{len(sub_df)} subjects, {len(diagnosis_df)} scans")

    return diagnosis_df, supplementary_diagnoses


def cleaning_nan_diagnoses(bids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Printing the number of missing diagnoses and filling it partially for ADNI datasets

    Parameters
    ----------
    bids_df: pd.DataFrame
        DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']

    Returns
    -------
    A cleaned DataFrame
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

    logger.info(f"Missing diagnoses: {missing_diag}")
    logger.info(f"Missing diagnoses not found: {missing_diag - found_diag}")

    return bids_copy_df


def df_to_tsv(tsv_path: Path, df: pd.DataFrame, baseline: bool = False) -> None:
    """
    Write Dataframe into a TSV file and drop duplicates

    Parameters
    ----------
    name: str
        Name of the tsv file
    results_path: str (path)
        Path to the folder
    df: DataFrame
        DataFrame you want to write in a TSV file.
        Columns must include ["participant_id", "session_id"].
    baseline: bool
        If True, there is only baseline session for each subject.
    """

    df.sort_values(by=["participant_id", "session_id"], inplace=True)
    if baseline:
        df.drop_duplicates(subset=["participant_id"], keep="first", inplace=True)
    else:
        df.drop_duplicates(
            subset=["participant_id", "session_id"], keep="first", inplace=True
        )
    df.to_csv(tsv_path, sep="\t", index=False)


def tsv_to_df(tsv_path: Path) -> pd.DataFrame:
    """
    Converts a TSV file to a Pandas DataFrame.

    Args:
        tsv_path (Path): Path to the TSV file to be read.

    Returns:
        pd.DataFrame: The resulting DataFrame containing the TSV data.

    Raises:
        ClinicaDLTSVError: If the TSV file cannot be found or is not in the correct format.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df = check_df(df)
    return df


def read_data(
    data: Union[str, Path, pd.DataFrame],
    check_protected_names: bool = True,
    check_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Reads an input dataframe, passed directly as a dataframe or via a path, and
    performs checks on it.

    Parameters
    ----------
    data : Union[str, Path, pd.DataFrame]
        The DataFrame as a pandas DataFrame or a path.

    Returns
    -------
    pd.DataFrame
        The dataframe, read and checked.

    Raises
    ------
    ValueError
        If 'data' is not a str, a Path or a pandas DataFrame.
    ClinicaDLTSVError
        If the DataFrame is empty.
    ClinicaDLTSVError
        If the required columns ('participant_id', 'session_id') are not found in the DataFrame.
    ClinicaDLTSVError
        If 'check_protected_names' is True and the dataframe contains columns named 'n_samples',
        'first_idx', 'last_idx' or 'dataset_id'.
    ClinicaDLTSVError
        If 'check_duplicates' is True and the dataframe contains duplicated (participant_id, session_id) pairs.
    """
    if isinstance(data, (str, Path)):
        data = Path(data)
        data = pd.read_csv(data, sep="\t")

    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f"'data' must be a path or a DataFrame. Got: {data}")

    return check_df(data, check_protected_names, check_duplicates)


def check_df(
    df: pd.DataFrame, check_protected_names: bool = True, check_duplicates: bool = True
) -> pd.DataFrame:
    """
    Checks the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, checked.

    Raises
    ------
    ClinicaDLTSVError
        If the DataFrame is empty.
    ClinicaDLTSVError
        If the required columns ('participant_id', 'session_id') are not found in the DataFrame.
    ClinicaDLTSVError
        If 'check_protected_names' is True and the dataframe contains columns named 'n_samples',
        'first_idx', 'last_idx' or 'dataset_id'.
    ClinicaDLTSVError
        If 'check_duplicates' is True and the dataframe contains duplicated (participant_id, session_id) pairs.
    """
    if len(df) == 0:
        raise ClinicaDLTSVError(f"The dataframe is empty!")

    if not {PARTICIPANT_ID, SESSION_ID}.issubset(set(df.columns.values)):
        raise ClinicaDLTSVError(
            f"The dataframe is not in the correct format. "
            f"Columns should include {PARTICIPANT_ID, SESSION_ID}"
        )
    if check_protected_names:
        protected_names = {N_SAMPLES, FIRST_INDEX, LAST_INDEX, DATASET_ID}
        if len(protected_names.intersection(set(df.columns.values))) > 0:
            raise ClinicaDLTSVError(
                f"The dataframe contains some protected column names. "
                f"Please do not use names in {protected_names}"
            )

    if check_duplicates:
        duplicated_pairs = df[df[[PARTICIPANT_ID, SESSION_ID]].duplicated(keep=False)]
        if len(duplicated_pairs) > 0:
            raise ClinicaDLTSVError(
                f"The dataframe contains duplicated (participant, session) pairs:\n"
                f"{duplicated_pairs}"
            )

    return df
