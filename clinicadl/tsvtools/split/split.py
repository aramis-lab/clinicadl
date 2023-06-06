# coding: utf8

from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
from sklearn.model_selection import StratifiedShuffleSplit

from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import (
    category_conversion,
    chi2,
    complementary_list,
    df_to_tsv,
    extract_baseline,
    find_label,
    remove_unicity,
    retrieve_longitudinal,
)

sex_dict = {"M": 0, "F": 1}
logger = getLogger("clinicadl.tsvtools.split")


def KStests(train_df, test_df, threshold=0.5):
    pmin = 1
    column = ""
    for col in train_df.columns:
        if col == "session_id":
            continue
        _, pval = ks_2samp(train_df[col], test_df[col])
        if pval < pmin:
            pmin = pval
            column = col
    return (pmin, column)


def shuffle_choice(df, n_shuffle=10):
    p_min_max, n_col_min = 0, df.columns.size

    for i in range(n_shuffle):
        train_df = df.sample(frac=0.75)
        test_df = df.drop(train_df.index)

        p, col = KStests(train_df, test_df)

        if p > p_min_max:
            p_min_max = p
            best_train_df, best_test_df = train_df, test_df

    return (best_train_df, best_test_df, p_min_max)


def KStests(train_df, test_df, threshold=0.5):
    pmin = 1
    column = ""
    for col in train_df.columns:
        if col == "session_id":
            continue
        _, pval = ks_2samp(train_df[col], test_df[col])
        if pval < pmin:
            pmin = pval
            column = col
    return (pmin, column)


def shuffle_choice(df, n_shuffle=10):
    p_min_max, n_col_min = 0, df.columns.size

    for i in range(n_shuffle):
        train_df = df.sample(frac=0.75)
        test_df = df.drop(train_df.index)

        p, col = KStests(train_df, test_df)

        if p > p_min_max:
            p_min_max = p
            best_train_df, best_test_df = train_df, test_df

    return (best_train_df, best_test_df, p_min_max)


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
    Split data at the subject-level in training and test set with equivalent age, sex and split_label distributions.

    Parameters
    ----------
    diagnosis_df: DataFrame
        Columns including ['participant_id', 'session_id', 'group']
    split_label: str
        Label on which the split is done (categorical variables)
    n_test: float
        If > 1 number of subjects to put in the test set.
        If < 1 proportion of subjects to put in the test set.
    p_age_threshold: float
        Threshold for the t-test on age.
    p_sex_threshold: float
        Threshold for the chi2 test on sex.
    supplementary_train_df: DataFrame
        Add data that must be included in the train set.
    ignore_demographics: bool
        If True the diagnoses are split without taking into account the demographics
        distributions (age, sex).

    Returns
    -------
    train_df: DataFrame
        Subjects in the train set
    test_df: DataFrame
        Subjects in the test set
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
    data_tsv: Path,
    n_test=100,
    subset_name="test",
    p_age_threshold=0.80,
    p_sex_threshold=0.80,
    categorical_split_variable=None,
    ignore_demographics=False,
    verbose=0,
    not_only_baseline=True,
    multi_diagnoses=False,
):
    """
    Performs a single split for each label independently on the subject level.
    There will be two TSV file for the train set (baseline and longitudinal),
    whereas there will only be one TSV file for the test set (baseline sessions).

    The age and sex distributions between the two sets must be non-significant (according to T-test and chi-square).

    Parameters
    ----------
    data_tsv: str (path)
        Path to the tsv containing data extracted by clinicadl tsvtools getlabels.
    n_test: float
        If >= 1, number of subjects to put in set with name 'subset_name'.
        If < 1, proportion of subjects to put in set with name 'subset_name'.
        If 0, no training set is created and the whole dataset is considered as one set with name 'subset_name'.
    subset_name: str
        Name of the subset that is complementary to train.
    p_age_threshold: float
        The threshold used for the T-test on age distributions.
    p_sex_threshold: float
        The threshold used for the T-test on sex distributions.
    categorical_split_variable: str
        Name of a categorical variable to perform a stratified split.
    ignore_demographics: bool
        If True the diagnoses are split without taking into account the demographics
        distributions (age, sex).
    verbose: int
        Level of verbosity.

    Informations
    ------------
    writes three files per <label>.tsv file present in data_tsv:
        - data_tsv/train/<label>.tsv
        - data_tsv/train/<label>_baseline.tsv
        - data_tsv/<subset_name>/<label>_baseline.tsv
    """

    parents_path = data_tsv.parents[0]
    split_numero = 1
    folder_name = f"split"

    while (parents_path / folder_name).is_dir():
        split_numero += 1
        folder_name = f"split_{split_numero}"
    results_path = parents_path / folder_name
    results_path.mkdir(parents=True)

    commandline_to_json(
        {
            "output_dir": results_path,
            "n_test": n_test,
            "subset_name": subset_name,
            "p_age_threshold": p_age_threshold,
            "p_sex_threshold": p_sex_threshold,
            "categorical_split_variable": categorical_split_variable,
            "ignore_demographics": ignore_demographics,
            "mullti_diagnoses": multi_diagnoses,
        },
        filename="split.json",
    )

    # The baseline session must be kept before or we are taking all the sessions to mix them

    if categorical_split_variable is None:
        categorical_split_variable = "diagnosis"
    else:
        categorical_split_variable.append("diagnosis")

    # Read files
    diagnosis_df_path = data_tsv.name
    diagnosis_df = pd.read_csv(data_tsv, sep="\t")
    list_columns = diagnosis_df.columns.values
    if multi_diagnoses:
        train, test, p_min = shuffle_choice(diagnosis_df, n_shuffle=5000)

        train_df = extract_baseline(train)
        test_df = extract_baseline(test)

        name = f"{subset_name}_baseline.tsv"
        df_to_tsv(name, results_path, test_df, baseline=True)

        if not_only_baseline:
            long_test_df = retrieve_longitudinal(test_df, diagnosis_df)
            name = f"{subset_name}.tsv"
            # long_test_df = long_test_df[["participant_id", "session_id"]]
            df_to_tsv(name, results_path, long_test_df)

    elif n_test > 0:
        if (
            "diagnosis" not in list_columns
            or ("age" not in list_columns and "age_bl" not in list_columns)
            or "sex" not in list_columns
        ):
            parents_path = parents_path.resolve()
            n = 0
            while not (parents_path / "labels.tsv").is_file() and n <= 4:
                parents_path = parents_path.parents[0]
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
                    f"The column 'age', 'sex' or 'diagnosis' is missing and the pipeline was not able to find "
                    "the output of clinicadl get-labels to get it."
                    "Please run clinicadl get-metadata to get these columns or add the the flag --ignore_demographics "
                    "to split without trying to balance age or sex distributions."
                )

        train_df, test_df = create_split(
            diagnosis_df,
            split_label=categorical_split_variable,
            n_test=n_test,
            p_age_threshold=p_age_threshold,
            p_sex_threshold=p_sex_threshold,
            ignore_demographics=ignore_demographics,
        )

        # train_df= train_df[["participant_id", "session_id"]]
        # test_df= test_df[["participant_id", "session_id"]]

        name = f"{subset_name}_baseline.tsv"
        df_to_tsv(name, results_path, test_df, baseline=True)

        if not_only_baseline:
            name = f"{subset_name}.tsv"
            long_test_df = retrieve_longitudinal(test_df, diagnosis_df)
            # long_test_df = long_test_df[["participant_id", "session_id"]]
            df_to_tsv(name, results_path, long_test_df)

    else:
        train_df = extract_baseline(diagnosis_df)
        # train_df = train_df[["participant_id", "session_id"]]
        if not_only_baseline:
            long_train_df = diagnosis_df

    name = "train_baseline.tsv"
    df_to_tsv(name, results_path, train_df, baseline=True)

    long_train_df = retrieve_longitudinal(train_df, diagnosis_df)
    # long_train_df = long_train_df[["participant_id", "session_id"]]
    name = "train.tsv"
    df_to_tsv(name, results_path, long_train_df)
