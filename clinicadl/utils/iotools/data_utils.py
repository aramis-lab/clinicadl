# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
from logging import getLogger
from pathlib import Path

import pandas as pd

from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)

logger = getLogger("clinicadl")


################################
# TSV files loaders
################################
def load_data_test(test_path: Path, diagnoses_list, baseline=True, multi_cohort=False):
    """
    Load data not managed by split_manager.

    Args:
        test_path (str): path to the test TSV files / split directory / TSV file for multi-cohort
        diagnoses_list (List[str]): list of the diagnoses wanted in case of split_dir or multi-cohort
        baseline (bool): If True baseline sessions only used (split_dir handling only).
        multi_cohort (bool): If True considers multi-cohort setting.
    """
    # TODO: computes baseline sessions on-the-fly to manager TSV file case

    if multi_cohort:
        if test_path.suffix != ".tsv":
            raise ClinicaDLArgumentError(
                "If multi_cohort is given, the TSV_DIRECTORY argument should be a path to a TSV file."
            )
        else:
            tsv_df = pd.read_csv(test_path, sep="\t")
            check_multi_cohort_tsv(tsv_df, "labels")
            test_df = pd.DataFrame()
            found_diagnoses = set()
            for idx in range(len(tsv_df)):
                cohort_name = tsv_df.loc[idx, "cohort"]
                cohort_path = Path(tsv_df.loc[idx, "path"])
                cohort_diagnoses = (
                    tsv_df.loc[idx, "diagnoses"].replace(" ", "").split(",")
                )
                if bool(set(cohort_diagnoses) & set(diagnoses_list)):
                    target_diagnoses = list(set(cohort_diagnoses) & set(diagnoses_list))
                    cohort_test_df = load_data_test_single(
                        cohort_path, target_diagnoses, baseline=baseline
                    )
                    cohort_test_df["cohort"] = cohort_name
                    test_df = pd.concat([test_df, cohort_test_df])
                    found_diagnoses = found_diagnoses | (
                        set(cohort_diagnoses) & set(diagnoses_list)
                    )

            if found_diagnoses != set(diagnoses_list):
                raise ValueError(
                    f"The diagnoses found in the multi cohort dataset {found_diagnoses} "
                    f"do not correspond to the diagnoses wanted {set(diagnoses_list)}."
                )
            test_df.reset_index(inplace=True, drop=True)
    else:
        if test_path.suffix == ".tsv":
            tsv_df = pd.read_csv(test_path, sep="\t")
            multi_col = {"cohort", "path"}
            if multi_col.issubset(tsv_df.columns.values):
                raise ClinicaDLConfigurationError(
                    "To use multi-cohort framework, please add 'multi_cohort=true' in your configuration file or '--multi_cohort' flag to the command line."
                )
        test_df = load_data_test_single(test_path, diagnoses_list, baseline=baseline)
        test_df["cohort"] = "single"

    return test_df


def check_test_path(test_path: Path, baseline: bool = True) -> Path:
    if baseline:
        train_filename = "train_baseline.tsv"
        label_filename = "labels_baseline.tsv"
    else:
        train_filename = "train.tsv"
        label_filename = "labels.tsv"

    if not (test_path.parent / train_filename).is_file():
        if not (test_path.parent / label_filename).is_file():
            raise ClinicaDLTSVError(
                f"There is no {train_filename} nor {label_filename} in your folder {test_path.parents[0]} "
            )
        else:
            test_path = test_path.parent / label_filename
    else:
        test_path = test_path.parent / train_filename

    return test_path


def load_data_test_single(test_path: Path, diagnoses_list, baseline=True):
    if test_path.suffix == ".tsv":
        test_df = pd.read_csv(test_path, sep="\t")
        if "diagnosis" not in test_df.columns.values:
            raise ClinicaDLTSVError(
                f"'diagnosis' column must be present in TSV file {test_path}."
            )
        test_df = test_df[test_df.diagnosis.isin(diagnoses_list)]
        if len(test_df) == 0:
            raise ClinicaDLTSVError(
                f"Diagnoses wanted {diagnoses_list} were not found in TSV file {test_path}."
            )
        return test_df

    test_path = check_test_path(test_path=test_path, baseline=baseline)
    test_df = pd.read_csv(test_path, sep="\t")
    test_df = test_df[test_df.diagnosis.isin(diagnoses_list)]
    test_df.reset_index(inplace=True, drop=True)

    return test_df


def check_multi_cohort_tsv(tsv_df: pd.DataFrame, purpose: str) -> None:
    """
    Checks that a multi-cohort TSV file is valid.

    Args:
        tsv_df (pd.DataFrame): DataFrame of multi-cohort definition.
        purpose (str): what the TSV file describes (CAPS or TSV).
    Raises:
        ValueError: if the TSV file is badly formatted.
    """
    mandatory_col = ("cohort", "diagnoses", "path")
    if purpose.upper() == "CAPS":
        mandatory_col = ("cohort", "path")
    if not set(mandatory_col).issubset(tsv_df.columns.values):
        raise ClinicaDLTSVError(
            f"Columns of the TSV file used for {purpose} location must include {mandatory_col}"
        )
