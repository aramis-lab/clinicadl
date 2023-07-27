# coding: utf-8

from copy import copy
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd

from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.tsvtools_utils import (
    add_demographics,
    cleaning_nan_diagnoses,
    find_label,
    first_session,
    merged_tsv_reader,
    next_session,
)

logger = getLogger("clinicadl.tsvtools.analysis")


def demographics_analysis(
    merged_tsv: Path, data_tsv: Path, results_tsv: Path, diagnoses
):
    """
    Produces a tsv file with rows corresponding to the labels defined by the diagnoses list,
    and the columns being demographic statistics.

    Writes one tsv file at results_tsv, containing the demographic analysis of the tsv files in data_tsv.

    Parameters
    ----------
    merged_tsv: str (path)
        Path to the file obtained by the command clinica iotools merge-tsv.
    data_tsv: str (path)
        Path to the folder containing data extracted by clinicadl tsvtool get-labels.
    results_tsv: str (path)
        Path to the output tsv file (filename included).
    diagnoses: list of str
        Labels selected for the demographic analysis.

    """

    if not data_tsv.is_file():
        raise ClinicaDLTSVError(f"{data_tsv} file was not found. ")

    if not merged_tsv.is_file():
        raise ClinicaDLTSVError(f"{merged_tsv} file was not found. ")
    merged_df = pd.read_csv(merged_tsv, sep="\t")
    merged_df.set_index(["participant_id", "session_id"], inplace=True)
    merged_df = cleaning_nan_diagnoses(merged_df)

    parent_directory = results_tsv.resolve().parent
    parent_directory.mkdir(parents=True, exist_ok=True)

    fields_dict = {
        "age": find_label(merged_df.columns.values, "age"),
        "sex": find_label(merged_df.columns.values, "sex"),
        "MMSE": find_label(merged_df.columns.values, "mms"),
        "CDR": "cdr_global",
    }

    columns = [
        "n_subjects",
        "mean_age",
        "std_age",
        "min_age",
        "max_age",
        "sexF",
        "sexM",
        "mean_MMSE",
        "std_MMSE",
        "min_MMSE",
        "max_MMSE",
        "CDR_0",
        "CDR_0.5",
        "CDR_1",
        "CDR_2",
        "CDR_3",
        "mean_scans",
        "std_scans",
        "n_scans",
    ]
    results_df = pd.DataFrame(
        index=diagnoses, columns=columns, data=np.zeros((len(diagnoses), len(columns)))
    )

    # Need all values for mean and variance (age, MMSE and scans)
    diagnosis_dict = dict.fromkeys(diagnoses)

    for diagnosis in diagnoses:
        diagnosis_dict[diagnosis] = {"age": [], "MMSE": [], "scans": []}
        getlabels_df = pd.read_csv(data_tsv, sep="\t")
        diagnosis_copy_df = copy(getlabels_df)
        diagnosis_copy_df = diagnosis_copy_df[
            diagnosis_copy_df["diagnosis"] == diagnosis
        ]
        if not diagnosis_copy_df.empty:
            diagnosis_demographics_df = add_demographics(
                diagnosis_copy_df, merged_df, diagnosis
            )
            diagnosis_demographics_df.reset_index()
            diagnosis_demographics_df.set_index(
                ["participant_id", "session_id"], inplace=True
            )
            diagnosis_copy_df.set_index(["participant_id", "session_id"], inplace=True)
            for subject, subject_df in diagnosis_copy_df.groupby(level=0):
                first_session_id = first_session(subject_df)
                feature_absence = isinstance(
                    merged_df.loc[(subject, first_session_id), "diagnosis"], float
                )
                while feature_absence:
                    first_session_id = next_session(subject_df, first_session_id)

                    feature_absence = isinstance(
                        merged_df.loc[(subject, first_session_id), "diagnosis"], float
                    )
                demographics_subject_df = merged_df.loc[subject]

                # Extract features
                logger.debug(f"extract features for subject {subject}")

                results_df.loc[diagnosis, "n_subjects"] += 1
                results_df.loc[diagnosis, "n_scans"] += len(subject_df)
                diagnosis_dict[diagnosis]["age"].append(
                    merged_df.loc[(subject, first_session_id), fields_dict["age"]]
                )
                diagnosis_dict[diagnosis]["MMSE"].append(
                    merged_df.loc[(subject, first_session_id), fields_dict["MMSE"]]
                )
                diagnosis_dict[diagnosis]["scans"].append(len(subject_df))
                sexF = (
                    len(
                        demographics_subject_df[
                            (demographics_subject_df[fields_dict["sex"]].isin(["F"]))
                        ]
                    )
                    > 0
                )
                sexM = (
                    len(
                        demographics_subject_df[
                            (demographics_subject_df[fields_dict["sex"]].isin(["M"]))
                        ]
                    )
                    > 0
                )
                if sexF:
                    results_df.loc[diagnosis, "sexF"] += 1
                elif sexM:
                    results_df.loc[diagnosis, "sexM"] += 1
                else:
                    raise ValueError(
                        f"The field 'sex' for patient {subject} can not be determined"
                    )

                cdr = merged_df.at[(subject, first_session_id), fields_dict["CDR"]]
                if cdr == 0:
                    results_df.loc[diagnosis, "CDR_0"] += 1
                elif cdr == 0.5:
                    results_df.loc[diagnosis, "CDR_0.5"] += 1
                elif cdr == 1:
                    results_df.loc[diagnosis, "CDR_1"] += 1
                elif cdr == 2:
                    results_df.loc[diagnosis, "CDR_2"] += 1
                elif cdr == 3:
                    results_df.loc[diagnosis, "CDR_3"] += 1
                else:
                    tt = 3  # warn(f"Patient {subject} has CDR {cdr}")
        else:
            raise ClinicaDLArgumentError(
                f"There is no subject with diagnosis {diagnosis}"
            )
    for diagnosis in diagnoses:
        logger.debug(f"compute stats for diagnosis {diagnosis}")

        results_df.loc[diagnosis, "mean_age"] = np.nanmean(
            diagnosis_dict[diagnosis]["age"]
        )
        results_df.loc[diagnosis, "std_age"] = np.nanstd(
            diagnosis_dict[diagnosis]["age"]
        )
        results_df.loc[diagnosis, "min_age"] = np.nanmin(
            diagnosis_dict[diagnosis]["age"]
        )
        results_df.loc[diagnosis, "max_age"] = np.nanmax(
            diagnosis_dict[diagnosis]["age"]
        )
        results_df.loc[diagnosis, "mean_MMSE"] = np.nanmean(
            diagnosis_dict[diagnosis]["MMSE"]
        )
        results_df.loc[diagnosis, "std_MMSE"] = np.nanstd(
            diagnosis_dict[diagnosis]["MMSE"]
        )
        results_df.loc[diagnosis, "min_MMSE"] = np.nanmin(
            diagnosis_dict[diagnosis]["MMSE"]
        )
        results_df.loc[diagnosis, "max_MMSE"] = np.nanmax(
            diagnosis_dict[diagnosis]["MMSE"]
        )
        results_df.loc[diagnosis, "mean_scans"] = np.nanmean(
            diagnosis_dict[diagnosis]["scans"]
        )
        results_df.loc[diagnosis, "std_scans"] = np.nanstd(
            diagnosis_dict[diagnosis]["scans"]
        )

        for key in diagnosis_dict[diagnosis]:
            if np.isnan(diagnosis_dict[diagnosis][key]).any():
                logger.warning(
                    f"NaN values were found for {key} values associated to diagnosis {diagnosis}"
                )

    results_df.index.name = "group"

    results_df.to_csv(results_tsv, sep="\t")
    logger.info(f"Result is stored at {results_tsv}")
