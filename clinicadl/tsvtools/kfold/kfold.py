# coding: utf8

import os
import shutil
from copy import copy
from logging import getLogger
from os import path
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import (
    extract_baseline,
    remove_sub_labels,
    retrieve_longitudinal,
)

sex_dict = {"M": 0, "F": 1}
logger = getLogger("clinicadl")


def write_splits(
    diagnosis_df: pd.DataFrame,
    split_label: str,
    n_splits: int,
    subset_name: str,
):
    """
    Split data at the subject-level in training and test to have equivalent distributions in split_label.
    Writes test and train Dataframes.

    Args:
        diagnosis: diagnosis on which the split is done
        diagnosis_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        split_label: label on which the split is done (categorical variables)
        n_splits: Number of splits in the k-fold cross-validation.
        train_path: Path to the training data.
        test_path: Path to the test data.
        supplementary_diagnoses: List of supplementary diagnoses to add to the data.
    """

    baseline_df = extract_baseline(diagnosis_df)
    # print(baseline_df)

    if split_label is None:
        diagnoses_list = list(baseline_df.group)
        unique = list(set(diagnoses_list))
        y = np.array([unique.index(x) for x in diagnoses_list])
    else:
        stratification_list = list(baseline_df[split_label])
        unique = list(set(stratification_list))
        y = np.array([unique.index(x) for x in stratification_list])

    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    output_train_df = pd.DataFrame()
    output_long_train_df = pd.DataFrame()
    output_all_df = pd.DataFrame()
    output_long_test_df = pd.DataFrame()

    # print(f"Label {diagnosis}")
    for i, indices in enumerate(splits.split(np.zeros(len(y)), y)):
        print(f"Split {i}")
        train_index, test_index = indices

        test_df = baseline_df.iloc[test_index]
        train_df = baseline_df.iloc[train_index]

        train_df = train_df.reindex(
            columns=train_df.columns.tolist() + ["split", "datagroup"]
        )
        train_df.__setitem__("split", int(i))
        train_df.__setitem__("datagroup", "train")
        test_df = test_df.reindex(
            columns=test_df.columns.tolist() + ["split", "datagroup"]
        )
        test_df.__setitem__("split", int(i))
        test_df.__setitem__("datagroup", subset_name)

        output_df = pd.concat([train_df, test_df])
        output_all_df = pd.concat([output_all_df, output_df])

        long_train_df = retrieve_longitudinal(train_df, diagnosis_df)
        long_train_df = long_train_df.reindex(
            columns=long_train_df.columns.tolist() + ["split", "datagroup"]
        )
        long_train_df.__setitem__("split", int(i))
        long_train_df.__setitem__("datagroup", "train")

        output_all_df = pd.concat([output_all_df, long_train_df])

    return output_all_df


def split_diagnoses(
    formatted_data_path: str,
    n_splits: int = 5,
    subset_name: str = "validation",
    stratification: str = None,
    test_tsv: str = None,
):
    """
    Performs a k-fold split for each label independently on the subject level.
    The train folder will contain two lists per fold per diagnosis (baseline and longitudinal),
    whereas the test folder will only include the list of baseline sessions for each spli.

    Writes three files per split per <label>.tsv file present in formatted_data_path:
            - formatted_data_path/train_splits-<n_splits>/split-<split>/<label>.tsv
            - formatted_data_path/train_splits-<n_splits>/split-<split>/<label>_baseline.tsv
            - formatted_data_path/<subset_name>_splits-<n_splits>/split-<split>/<label>_baseline.tsv

    Args:
        formatted_data_path: Path to the folder containing data extracted by clinicadl tsvtool getlabels.
        n_splits: Number of splits in the k-fold cross-validation.
        subset_name: Name of the subset that is complementary to train.
        MCI_sub_categories: If True, manages MCI sub-categories to avoid data leakage.
        stratification: Name of variable used to stratify k-fold.
    """

    results_path = Path(formatted_data_path).parents[0]

    commandline_to_json(
        {
            "output_dir": results_path,
            "n_splits": n_splits,
            "subset_name": subset_name,
            "stratification": stratification,
        },
        filename="kfold.json",
    )

    # Read files

    # diagnosis_df_path=Path(formatted_data_path).name
    diagnosis_df = pd.read_csv(formatted_data_path, sep="\t")

    if test_tsv is not None:
        test_df = pd.read_csv(test_tsv, sep="\t")
        subjects_list = pd.unique(test_df["participant_id"])
        for subject in diagnosis_df.index.values:
            if subject in subjects_list:
                diagnosis_df.drop(subject)
    output_df = pd.DataFrame()

    # The baseline session must be kept before or we are taking all the sessions to mix them
    for diagnosis in pd.unique(diagnosis_df["group"]):
        diagnosis_copy_df = copy(diagnosis_df)

        indexName = diagnosis_copy_df[(diagnosis_copy_df["group"] != diagnosis)].index
        diagnosis_copy_df.drop(indexName, inplace=True)
        temp_df = write_splits(diagnosis_copy_df, stratification, n_splits, subset_name)
        temp_df.drop_duplicates(keep="first", inplace=True)
        output_df = pd.concat([output_df, temp_df])

    output_df = output_df.sort_values(["participant_id", "session_id", "split"])
    output_df.to_csv(
        path.join(results_path, f"{subset_name}.tsv"),
        sep="\t",
        index=False,
    )

    logger.info(f"K-fold split is done.")
