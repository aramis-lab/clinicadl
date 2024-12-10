import json
from logging import getLogger
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from scipy.stats import ks_2samp, ttest_ind
from sklearn.model_selection import StratifiedShuffleSplit

from clinicadl.dataset.utils import tsv_to_df
from clinicadl.splitter.make_splits.utils import (
    _write_to_csv,
    preprocess_stratification,
)
from clinicadl.splitter.splitter.single_split import SingleSplitConfig
from clinicadl.tsvtools.tsvtools_utils import extract_baseline, retrieve_longitudinal

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"

# coding: utf8


sex_dict = {"M": 0, "F": 1}
logger = getLogger("clinicadl.tsvtools.split")


def complementary_list(total_list, sub_list):
    result_list = []
    for element in total_list:
        if element not in sub_list:
            result_list.append(element)
    return result_list


def chi2(x_test, x_train):
    from scipy.stats import chisquare

    # Look for chi2 computation
    total_categories = np.concatenate([x_test, x_train])
    unique_categories = np.unique(total_categories)
    f_obs = [(x_test == category).sum() / len(x_test) for category in unique_categories]
    f_exp = [
        (x_train == category).sum() / len(x_train) for category in unique_categories
    ]
    T, p = chisquare(f_obs, f_exp)

    return T, p


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


def ks_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Perform Kolmogorov-Smirnov tests on all columns except session_id.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame.
    test_df : pd.DataFrame
        Testing DataFrame.

    Returns
    -------
    tuple
        The minimum p-value and the corresponding column name.
    """
    p_min = 1
    selected_column = ""
    for col in train_df.columns:
        if col == SESSION_ID:
            continue
        _, p_val = ks_2samp(train_df[col], test_df[col])
        if p_val < p_min:  # type: ignore
            p_min = p_val
            selected_column = col
    return p_min, selected_column


def shuffle_split_choice(df: pd.DataFrame, n_shuffle: int = 10) -> tuple:
    """
    Perform multiple shuffles and find the split with the highest p-value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_shuffle : int
        Number of shuffles.

    Returns
    -------
    tuple
        The best training DataFrame, test DataFrame, and p-value.
    """
    best_p_value = 0
    best_train_df, best_test_df = None, None

    for _ in range(n_shuffle):
        train_df = df.sample(frac=0.75, random_state=np.random.randint(1000))
        test_df = df.drop(train_df.index)

        p_value, _ = ks_test(train_df, test_df)

        if p_value > best_p_value:
            best_p_value = p_value
            best_train_df, best_test_df = train_df, test_df

    return best_train_df, best_test_df, best_p_value


def find_age_and_sex_values(train_index, test_index, age, sex):
    if len(set(age)) != 1:
        age_test = [float(age[idx]) for idx in test_index]
        age_train = [float(age[idx]) for idx in train_index]
        _, p_age = ttest_ind(age_test, age_train, nan_policy="omit")
    else:
        p_age = 1

    if len(set(sex)) != 1:
        sex_test = [sex_dict[sex[idx]] for idx in test_index]
        sex_train = [sex_dict[sex[idx]] for idx in train_index]
        _, p_sex = chi2(sex_test, sex_train)
    else:
        p_sex = 1

    logger.info(f"p_age={p_age:.2f}, p_sex={p_sex:.4f}")

    return p_age, p_sex


def make_split(
    tsv_path: Path,
    output_dir: Optional[Path] = None,
    n_test: PositiveFloat = 100,
    subset_name: str = "test",
    p_sex_threshold: float = 0.80,
    p_age_threshold: float = 0.80,
    stratification: Optional[List[str]] = None,
    ignore_demographics: bool = False,
    valid_longitudinal=False,
):
    """
    Performs a single split for each label independently on the subject level.
    There will be two TSV file for the train set (baseline and longitudinal),
    whereas there will only be one TSV file for the test set (baseline sessions).

    The age and sex distributions between the two sets must be non-significant (according to T-test and chi-square).

    """

    if not output_dir:
        output_dir = tsv_path.parent

    config = SingleSplitConfig(
        split_dir=output_dir,
        subset_name=subset_name,
        valid_longitudinal=valid_longitudinal,
        n_test=n_test,
        p_age_threshold=p_age_threshold,
        p_sex_threshold=p_sex_threshold,
        stratification=stratification,
        ignore_demographics=ignore_demographics,
    )

    config._check_split_dir()
    config._write_json()

    df = tsv_to_df(tsv_path)
    baseline_df = extract_baseline(df)

    stratify_labels = preprocess_stratification(
        df=baseline_df,
        columns=config.stratification,
        ignore_demographics=config.ignore_demographics,
    )

    # split_dir = config.split_dir / "split"
    # split_dir.mkdir(parents=True, exist_ok=True)

    if n_test > 0:
        n_test = int(n_test) if n_test >= 1 else int(n_test * len(baseline_df))

        if not ignore_demographics:
            sex = list(baseline_df["sex"].values)
            age = list(baseline_df["age"].values)

            flag_selection = True
            n_try = 0

            while flag_selection:
                splits = StratifiedShuffleSplit(n_splits=1, test_size=n_test)

                for train_index, test_index in splits.split(
                    baseline_df, stratify_labels
                ):
                    p_age, p_sex = find_age_and_sex_values(
                        train_index, test_index, age, sex
                    )

                    if p_sex >= p_sex_threshold and p_age >= p_age_threshold:
                        flag_selection = False
                        test_df = baseline_df.loc[test_index]
                        train_df = baseline_df.loc[train_index]

                    n_try += 1
            logger.info(f"Split was found after {n_try} trials.")

        else:
            idx = np.arange(len(baseline_df))
            idx_test = np.sort(np.random.choice(idx, size=n_test, replace=False))
            idx_train = complementary_list(idx, idx_test)

            test_df = baseline_df.loc[idx_test]
            train_df = baseline_df.loc[idx_train]

        _write_to_csv(test_df, config.split_dir / f"{config.subset_name}_baseline.tsv")
        if valid_longitudinal:
            long_test_df = retrieve_longitudinal(test_df, df)
            _write_to_csv(long_test_df, config.split_dir / f"{subset_name}.tsv")

    else:
        train_df = baseline_df

    _write_to_csv(train_df, config.split_dir / "train_baseline.tsv")

    long_train_df = retrieve_longitudinal(train_df, df)
    _write_to_csv(long_train_df, config.split_dir / "train.tsv")

    return config.split_dir
