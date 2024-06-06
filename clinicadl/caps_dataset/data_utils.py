# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
import abc
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from clinicadl.caps_dataset.data import (
    CapsDataset,
    CapsDatasetImage,
    CapsDatasetPatch,
    CapsDatasetRoi,
    CapsDatasetSlice,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)

logger = getLogger("clinicadl")


def return_dataset(
    input_dir: Path,
    data_df: pd.DataFrame,
    preprocessing_dict: Dict[str, Any],
    all_transformations: Optional[Callable],
    label: str = None,
    label_code: Dict[str, int] = None,
    train_transformations: Optional[Callable] = None,
    cnn_index: int = None,
    label_presence: bool = True,
    multi_cohort: bool = False,
) -> CapsDataset:
    """
    Return appropriate Dataset according to given options.
    Args:
        input_dir: path to a directory containing a CAPS structure.
        data_df: List subjects, sessions and diagnoses.
        preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
        train_transformations: Optional transform to be applied during training only.
        all_transformations: Optional transform to be applied during training and evaluation.
        label: Name of the column in data_df containing the label.
        label_code: label code that links the output node number to label value.
        cnn_index: Index of the CNN in a multi-CNN paradigm (optional).
        label_presence: If True the diagnosis will be extracted from the given DataFrame.
        multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

    Returns:
         the corresponding dataset.
    """
    if cnn_index is not None and preprocessing_dict["mode"] == "image":
        raise NotImplementedError(
            f"Multi-CNN is not implemented for {preprocessing_dict['mode']} mode."
        )

    if preprocessing_dict["mode"] == "image":
        return CapsDatasetImage(
            input_dir,
            data_df,
            preprocessing_dict,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif preprocessing_dict["mode"] == "patch":
        return CapsDatasetPatch(
            input_dir,
            data_df,
            preprocessing_dict,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            patch_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif preprocessing_dict["mode"] == "roi":
        return CapsDatasetRoi(
            input_dir,
            data_df,
            preprocessing_dict,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            roi_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif preprocessing_dict["mode"] == "slice":
        return CapsDatasetSlice(
            input_dir,
            data_df,
            preprocessing_dict,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            slice_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    else:
        raise NotImplementedError(
            f"Mode {preprocessing_dict['mode']} is not implemented."
        )


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
        if not test_path.suffix == ".tsv":
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

    test_df = pd.DataFrame()

    if baseline:
        if not (test_path.parent / "train_baseline.tsv").is_file():
            if not (test_path.parent / "labels_baseline.tsv").is_file():
                raise ClinicaDLTSVError(
                    f"There is no train_baseline.tsv nor labels_baseline.tsv in your folder {test_path.parents[0]} "
                )
            else:
                test_path = test_path.parent / "labels_baseline.tsv"
        else:
            test_path = test_path.parent / "train_baseline.tsv"
    else:
        if not (test_path.parent / "train.tsv").is_file():
            if not (test_path.parent / "labels.tsv").is_file():
                raise ClinicaDLTSVError(
                    f"There is no train.tsv or labels.tsv in your folder {test_path.parent} "
                )
            else:
                test_path = test_path.parent / "labels.tsv"
        else:
            test_path = test_path.parent / "train.tsv"

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
