# coding: utf8

import os
import shutil
from copy import copy
from logging import getLogger
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedShuffleSplit

from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import (
    category_conversion,
    chi2,
    complementary_list,
    extract_baseline,
    find_label,
    remove_sub_labels,
    remove_unicity,
    retrieve_longitudinal,
)

sex_dict = {"M": 0, "F": 1}
logger = getLogger("clinicadl")


def df_to_tsv(name: str, results_path: str, df, baseline: bool = False):

    df = df[["participant_id", "session_id"]]
    df.sort_values(by=["participant_id", "session_id"], inplace=True)
    if baseline:
        df.drop_duplicates(subset=["participant_id"], keep="first", inplace=True)
    else:
        df.drop_duplicates(
            subset=["participant_id", "session_id"], keep="first", inplace=True
        )
    df.to_csv(path.join(results_path, name), sep="\t", index=False)


def create_split(
    diagnosis_df,
    split_label,
    n_test,
    p_age_threshold=0.80,
    p_sex_threshold=0.80,
    supplementary_train_df=None,
    ignore_demographics=False,
):

    """
    Split data at the subject-level in training and test set with equivalent age, sex and split_label distributions

    Args:
        diagnosis_df: DataFrame with columns including ['participant_id', 'session_id', 'group']
        split_label: (str) label on which the split is done (categorical variables)
        n_test: (float)
            If > 1 number of subjects to put in the test set.
            If < 1 proportion of subjects to put in the test set.
        p_age_threshold: (float) threshold for the t-test on age.
        p_sex_threshold: (float) threshold for the chi2 test on sex.
        supplementary_train_df: (DataFrame) Add data that must be included in the train set.
        ignore_demographics: (bool): If True the diagnoses are split without taking into account the demographics
            distributions (age, sex).

    Returns:
        train_df (DataFrame) subjects in the train set
        test_df (DataFrame) subjects in the test set
    """
    if supplementary_train_df is not None:
        sup_train_sex = [sex_dict[x] for x in supplementary_train_df.sex.values]
        sup_train_age = [float(x) for x in supplementary_train_df.age.values]
    else:
        sup_train_sex = []
        sup_train_age = []

    baseline_df = extract_baseline(diagnosis_df)
    if n_test >= 1:
        n_test = int(n_test)
    else:
        n_test = int(n_test * len(baseline_df))

    if not {split_label}.issubset(set(baseline_df.columns.values)):
        raise ClinicaDLArgumentError(
            f"The column {split_label} is missing."
            f"Please add it using the --variables_of_interest flag in getlabels."
        )

    if not ignore_demographics:
        try:
            sex_label = find_label(baseline_df.columns.values, "sex")
            age_label = find_label(baseline_df.columns.values, "age")
        except ClinicaDLArgumentError:
            raise ClinicaDLArgumentError(
                "This dataset do not have age or sex values. "
                "Please add the flag --ignore_demographics to split "
                "without trying to balance age or sex distributions."
            )

        sex = list(baseline_df[sex_label].values)
        age = list(baseline_df[age_label].values)
        category = list(baseline_df[split_label].values)
        category = category_conversion(category)
        category = remove_unicity(category)

        flag_selection = True
        n_try = 0

        while flag_selection:

            splits = StratifiedShuffleSplit(n_splits=1, test_size=n_test)
            for train_index, test_index in splits.split(category, category):

                # Find the value for different demographics (age & sex)
                if len(set(age)) != 1:
                    age_test = [float(age[idx]) for idx in test_index]
                    age_train = [float(age[idx]) for idx in train_index] + sup_train_age
                    _, p_age = ttest_ind(age_test, age_train, nan_policy="omit")
                else:
                    p_age = 1

                if len(set(sex)) != 1:
                    sex_test = [sex_dict[sex[idx]] for idx in test_index]
                    sex_train = [
                        sex_dict[sex[idx]] for idx in train_index
                    ] + sup_train_sex
                    _, p_sex = chi2(sex_test, sex_train)
                else:
                    p_sex = 1

                logger.info(f"p_age={p_age:.2f}, p_sex={p_sex:.4f}")

                if p_sex >= p_sex_threshold and p_age >= p_age_threshold:
                    flag_selection = False
                    test_df = baseline_df.loc[test_index]
                    train_df = baseline_df.loc[train_index]
                    if supplementary_train_df is not None:
                        train_df = pd.concat([train_df, supplementary_train_df])
                        train_df.reset_index(drop=True, inplace=True)

                n_try += 1
        logger.info(f"Split was found after {n_try} trials.")

    else:
        idx = np.arange(len(baseline_df))
        idx_test = np.random.choice(idx, size=n_test, replace=False)
        idx_test.sort()
        idx_train = complementary_list(idx, idx_test)
        test_df = baseline_df.loc[idx_test]
        train_df = baseline_df.loc[idx_train]

    return train_df, test_df


def split_diagnoses(
    formatted_data_path,
    n_test=100,
    subset_name="test",
    p_age_threshold=0.80,
    p_sex_threshold=0.80,
    categorical_split_variable=None,
    ignore_demographics=False,
    verbose=0,
    not_only_baseline=True,
):
    """
    Performs a single split for each label independently on the subject level.
    There will be two TSV file for the train set (baseline and longitudinal),
    whereas there will only be one TSV file for the test set (baseline sessions).

    The age and sex distributions between the two sets must be non-significant (according to T-test and chi-square).

    Args:
        formatted_data_path (str): Path to the tsv containing data extracted by clinicadl tsvtools getlabels.
        n_test (float):
            If >= 1, number of subjects to put in set with name 'subset_name'.
            If < 1, proportion of subjects to put in set with name 'subset_name'.
            If 0, no training set is created and the whole dataset is considered as one set with name 'subset_name'.
        subset_name (str): Name of the subset that is complementary to train.
        p_age_threshold (float): The threshold used for the T-test on age distributions.
        p_sex_threshold (float): The threshold used for the T-test on sex distributions.
        categorical_split_variable (str): name of a categorical variable to perform a stratified split.
        ignore_demographics (bool): If True the diagnoses are split without taking into account the demographics
            distributions (age, sex).
        verbose (int): level of verbosity.

    Returns:
        writes three files per <label>.tsv file present in formatted_data_path:
            - formatted_data_path/train/<label>.tsv
            - formatted_data_path/train/<label>_baseline.tsv
            - formatted_data_path/<subset_name>/<label>_baseline.tsv
    """

    results_path = Path(formatted_data_path).parents[0]

    commandline_to_json(
        {
            "output_dir": results_path,
            "n_test": n_test,
            "subset_name": subset_name,
            "p_age_threshold": p_age_threshold,
            "p_sex_threshold": p_sex_threshold,
            "categorical_split_variable": categorical_split_variable,
            "ignore_demographics": ignore_demographics,
        },
        filename="split.json",
    )

    # The baseline session must be kept before or we are taking all the sessions to mix them

    if categorical_split_variable is None:
        categorical_split_variable = "group"
    else:
        categorical_split_variable.append("group")

    # Read files
    diagnosis_df_path = Path(formatted_data_path).name
    diagnosis_df = pd.read_csv(formatted_data_path, sep="\t")

    if n_test > 0:

        train_df, test_df = create_split(
            diagnosis_df,
            split_label="diagnosis",
            n_test=n_test,
            p_age_threshold=p_age_threshold,
            p_sex_threshold=p_sex_threshold,
            ignore_demographics=ignore_demographics,
        )

        name = f"{subset_name}_baseline.tsv"
        df_to_tsv(name, results_path, test_df, baseline=True)

        if not_only_baseline:
            name = f"{subset_name}.tsv"
            long_test_df = retrieve_longitudinal(test_df, diagnosis_df)
            df_to_tsv(name, results_path, long_test_df)

    else:
        output_train_df = extract_baseline(diagnosis_copy_df)
        output_long_train_df = diagnosis_copy_df

    name = "train_baseline.tsv"
    df_to_tsv(name, results_path, train_df, baseline=True)

    long_train_df = retrieve_longitudinal(train_df, diagnosis_df)
    name = "train.tsv"
    df_to_tsv(name, results_path, long_train_df)
