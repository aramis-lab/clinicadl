# coding: utf8

import os
import shutil
from logging import getLogger
from os import path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedShuffleSplit

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


def create_split(
    diagnosis,
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
        diagnosis: (str) diagnosis on which the split is done
        diagnosis_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
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
        raise ValueError(
            f"The column {split_label} is missing."
            f"Please add it using the --variables_of_interest flag in getlabels."
        )

    if not ignore_demographics:
        try:
            sex_label = find_label(baseline_df.columns.values, "sex")
            age_label = find_label(baseline_df.columns.values, "age")
        except ValueError:
            raise ValueError(
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

                logger.debug("p_age=%.2f, p_sex=%.4f" % (p_age, p_sex))

                if p_sex >= p_sex_threshold and p_age >= p_age_threshold:
                    flag_selection = False
                    test_df = baseline_df.loc[test_index]
                    train_df = baseline_df.loc[train_index]
                    if supplementary_train_df is not None:
                        train_df = pd.concat([train_df, supplementary_train_df])
                        train_df.reset_index(drop=True, inplace=True)

                n_try += 1
        logger.info(f"Split for diagnosis {diagnosis} was found after {n_try} trials.")

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
    MCI_sub_categories=True,
    p_age_threshold=0.80,
    p_sex_threshold=0.80,
    categorical_split_variable=None,
    ignore_demographics=False,
    verbose=0,
):
    """
    Performs a single split for each label independently on the subject level.
    The train folder will contain two lists per diagnosis (baseline and longitudinal),
    whereas the test folder will only include the list of baseline sessions.

    The age and sex distributions between the two sets must be non-significant (according to T-test and chi-square).

    Args:
        formatted_data_path (str): Path to the folder containing data extracted by clinicadl tsvtool getlabels.
        n_test (float):
            If >= 1, number of subjects to put in set with name 'subset_name'.
            If < 1, proportion of subjects to put in set with name 'subset_name'.
            If 0, no training set is created and the whole dataset is considered as one set with name 'subset_name'.
        subset_name (str): Name of the subset that is complementary to train.
        MCI_sub_categories (bool): If True, manages MCI sub-categories to avoid data leakage.
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
    commandline_to_json(
        {
            "output_dir": formatted_data_path,
            "n_test": n_test,
            "subset_name": subset_name,
            "MCI_sub_categories": MCI_sub_categories,
            "p_age_threshold": p_age_threshold,
            "p_sex_threshold": p_sex_threshold,
            "categorical_split_variable": categorical_split_variable,
            "ignore_demographics": ignore_demographics,
        },
        filename="split.json",
    )

    # Read files
    results_path = formatted_data_path

    train_path = path.join(results_path, "train")
    if path.exists(train_path):
        shutil.rmtree(train_path)
    if n_test > 0:
        os.makedirs(train_path)

    if categorical_split_variable is None:
        categorical_split_variable = "diagnosis"

    test_path = path.join(results_path, subset_name)
    if path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)

    diagnosis_df_paths = os.listdir(results_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith(".tsv")]
    diagnosis_df_paths = [
        x for x in diagnosis_df_paths if not x.endswith("_baseline.tsv")
    ]

    MCI_special_treatment = False

    if "MCI.tsv" in diagnosis_df_paths and n_test > 0:
        if MCI_sub_categories:
            diagnosis_df_paths.remove("MCI.tsv")
            MCI_special_treatment = True
        elif "sMCI.tsv" in diagnosis_df_paths or "pMCI.tsv" in diagnosis_df_paths:
            logger.warning(
                "MCI special treatment was deactivated though MCI subgroups were found."
                "Be aware that it may cause data leakage in transfer learning tasks."
            )

    # The baseline session must be kept before or we are taking all the sessions to mix them
    for diagnosis_df_path in diagnosis_df_paths:
        diagnosis_df = pd.read_csv(path.join(results_path, diagnosis_df_path), sep="\t")
        interest_columns = diagnosis_df.columns.values
        diagnosis = diagnosis_df_path.split(".")[0]
        logger.info(f"Running split for diagnosis {diagnosis}")
        if n_test > 0:
            train_df, test_df = create_split(
                diagnosis,
                diagnosis_df,
                categorical_split_variable,
                n_test=n_test,
                p_age_threshold=p_age_threshold,
                p_sex_threshold=p_sex_threshold,
                ignore_demographics=ignore_demographics,
            )
            # Save baseline splits
            train_df.to_csv(
                path.join(train_path, f"{diagnosis}_baseline.tsv"),
                sep="\t",
                index=False,
            )
            test_df.to_csv(
                path.join(test_path, f"{diagnosis}_baseline.tsv"), sep="\t", index=False
            )

            long_train_df = retrieve_longitudinal(train_df, diagnosis_df)
            long_train_df.to_csv(
                path.join(train_path, f"{diagnosis}.tsv"), sep="\t", index=False
            )
            long_test_df = retrieve_longitudinal(test_df, diagnosis_df)
            long_test_df.to_csv(
                path.join(test_path, f"{diagnosis}.tsv"), sep="\t", index=False
            )

        else:
            baseline_df = extract_baseline(diagnosis_df)
            test_df = baseline_df[interest_columns]
            test_df.to_csv(
                path.join(test_path, f"{diagnosis}_baseline.tsv"), sep="\t", index=False
            )
            long_test_df = retrieve_longitudinal(test_df, diagnosis_df)
            long_test_df.to_csv(
                path.join(test_path, f"{diagnosis}.tsv"), sep="\t", index=False
            )

    if MCI_special_treatment:

        # Extraction of MCI subjects without intersection with the sMCI / pMCI train
        diagnosis_df = pd.read_csv(path.join(results_path, "MCI.tsv"), sep="\t")
        MCI_df = diagnosis_df.set_index(["participant_id", "session_id"])
        baseline_df = extract_baseline(MCI_df, set_index=False)

        if n_test > 1:
            n_test = int(n_test)
        else:
            n_test = int(n_test * len(baseline_df))

        MCI_df, supplementary_diagnoses = remove_sub_labels(
            MCI_df, ["sMCI", "pMCI"], diagnosis_df_paths, results_path
        )
        if len(supplementary_diagnoses) == 0:
            raise ValueError(
                "The MCI_sub_categories flag is not needed as there are no intersections with"
                "MCI subcategories."
            )

        # Construction of supplementary train
        supplementary_train_df = pd.DataFrame()
        for diagnosis in supplementary_diagnoses:
            sup_baseline_train_df = pd.read_csv(
                path.join(train_path, f"{diagnosis}_baseline.tsv"), sep="\t"
            )
            supplementary_train_df = pd.concat(
                [supplementary_train_df, sup_baseline_train_df]
            )
            sub_df = (
                supplementary_train_df.reset_index()
                .groupby("participant_id")["session_id"]
                .nunique()
            )
            logger.debug(
                f"supplementary_train_df {len(sub_df)} subjects, {len(supplementary_diagnoses)} scans"
            )

        supplementary_train_df.reset_index(drop=True, inplace=True)

        # MCI selection
        MCI_df.reset_index(inplace=True)
        baseline_df = extract_baseline(MCI_df)

        train_df, test_df = create_split(
            "MCI",
            baseline_df,
            categorical_split_variable,
            n_test=n_test,
            p_age_threshold=p_age_threshold,
            p_sex_threshold=p_sex_threshold,
            ignore_demographics=ignore_demographics,
            supplementary_train_df=supplementary_train_df,
        )

        # Write selection of MCI
        train_df.to_csv(
            path.join(train_path, "MCI_baseline.tsv"), sep="\t", index=False
        )
        test_df.to_csv(path.join(test_path, "MCI_baseline.tsv"), sep="\t", index=False)

        long_train_df = retrieve_longitudinal(train_df, diagnosis_df)
        long_train_df.to_csv(path.join(train_path, "MCI.tsv"), sep="\t", index=False)
        long_test_df = retrieve_longitudinal(test_df, diagnosis_df)
        long_test_df.to_csv(path.join(test_path, "MCI.tsv"), sep="\t", index=False)
