# coding: utf8

from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from clinicadl.utils.exceptions import ClinicaDLTSVError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import extract_baseline, retrieve_longitudinal

sex_dict = {"M": 0, "F": 1}
logger = getLogger("clinicadl.tsvtools.kfold")


def write_splits(
    diagnosis_df: pd.DataFrame,
    split_label: str,
    n_splits: int,
    subset_name: str,
    results_directory: Path,
):
    """
    Split data at the subject-level in training and test to have equivalent distributions in split_label.
    Writes test and train Dataframes.

    Parameters
    ----------
    diagnosis_df: Dataframe
        Columns must include ['participant_id', 'session_id', 'diagnosis']
    split_label: str
        Label on which the split is done (categorical variables)
    n_splits: int
        Number of splits in the k-fold cross-validation.
    subset_name: str
        Name of the subset split.
    results_directory: str (path)
        Path to the results directory.

    """

    baseline_df = extract_baseline(diagnosis_df)

    if split_label is None:
        diagnoses_list = list(baseline_df["diagnosis"])
        unique = list(set(diagnoses_list))
        y = np.array([unique.index(x) for x in diagnoses_list])
    else:
        stratification_list = list(baseline_df[split_label])
        unique = list(set(stratification_list))
        y = np.array([unique.index(x) for x in stratification_list])

    splits = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=2)

    for i, indices in enumerate(splits.split(np.zeros(len(y)), y)):
        train_index, test_index = indices

        test_df = baseline_df.iloc[test_index]
        train_df = baseline_df.iloc[train_index]
        long_train_df = retrieve_longitudinal(train_df, diagnosis_df)

        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        # train_df = train_df[["participant_id", "session_id"]]
        # test_df = test_df[["participant_id", "session_id"]]
        # long_train_df = long_train_df[["participant_id", "session_id"]]

        (results_directory / f"split-{i}").mkdir(parents=True)

        train_df.to_csv(
            results_directory / f"split-{i}" / "train_baseline.tsv",
            sep="\t",
            index=False,
        )
        test_df.to_csv(
            results_directory / f"split-{i}" / f"{subset_name}_baseline.tsv",
            sep="\t",
            index=False,
        )

        long_train_df.to_csv(
            results_directory / f"split-{i}" / "train.tsv",
            sep="\t",
            index=False,
        )


def split_diagnoses(
    data_tsv: Path,
    n_splits: int = 5,
    subset_name: str = None,
    stratification: str = None,
    merged_tsv: Path = None,
):
    """
    Performs a k-fold split for each label independently on the subject level.
    The output (the tsv file) will have two new columns :
        - split, with the number of the split the subject is in.
        - datagroup, with the name of the group (train or subset_name) the subject is in.

    The train group will contain baseline and longitudinal sessions,
    whereas the test group will only include the baseline sessions for each split.

    Parameters
    ----------
    data_tsv: str (path)
        Path to the tsv file extracted by clinicadl tsvtool getlabels.
    n_splits: int
        Number of splits in the k-fold cross-validation.
    subset_name: str
        Name of the subset that is complementary to train.
    stratification: str
        Name of variable used to stratify k-fold.
    merged_tsv: str
        Path to the merged.tsv file, output of clinica iotools merge-tsv.
    """

    parents_path = data_tsv.parent
    split_numero = 1
    folder_name = f"{n_splits}_fold"

    while (parents_path / folder_name).is_dir():
        split_numero += 1
        folder_name = f"{n_splits}_fold_{split_numero}"
    results_directory = parents_path / folder_name
    results_directory.mkdir(parents=True)

    commandline_to_json(
        {
            "output_dir": results_directory,
            "n_splits": n_splits,
            "subset_name": subset_name,
            "stratification": stratification,
        },
        filename="kfold.json",
    )

    diagnosis_df = pd.read_csv(data_tsv, sep="\t")
    list_columns = diagnosis_df.columns.values
    if (
        "diagnosis" not in list_columns
        or ("age" not in list_columns and "age_bl" not in list_columns)
        or "sex" not in list_columns
    ):
        logger.debug("Looking for the missing columns in others files.")
        if merged_tsv is None:
            parents_path = parents_path.resolve()
            n = 0
            while not (parents_path / "labels.tsv").is_file() and n <= 4:
                parents_path = parents_path.parent
                n += 1
            try:
                labels_df = pd.read_csv(parents_path / "labels.tsv", sep="\t")
                diagnosis_df = pd.merge(
                    diagnosis_df,
                    labels_df,
                    how="inner",
                    on=["participant_id", "session_id"],
                )
            except:
                raise ClinicaDLTSVError(
                    f"Your tsv file doesn't contain one of these columns : age, sex, diagnosis "
                    "and the pipeline wasn't able to find the output of clinicadl get-labels to get it."
                    "Before running this pipeline again, please run the command clinicadl get-metadata to get the missing columns"
                    "or add the the flag --ignore_demographics to split without trying to balance age or sex distributions."
                    "or add the option --merged-tsv to give the path the output of clinica merge-tsv"
                )
        else:
            labels_df = pd.read_csv(merged_tsv, sep="\t")
            diagnosis_df = pd.merge(
                diagnosis_df,
                labels_df,
                how="inner",
                on=["participant_id", "session_id"],
            )
    write_splits(diagnosis_df, stratification, n_splits, subset_name, results_directory)

    logger.info(f"K-fold split is done.")
