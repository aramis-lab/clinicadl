from logging import getLogger
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import NonNegativeInt, PositiveFloat
from scipy.stats import ks_2samp, ttest_ind
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from clinicadl.dataset.utils import tsv_to_df
from clinicadl.tsvtools.tsvtools_utils import (
    category_conversion,
    chi2,
    complementary_list,
    df_to_tsv,
    extract_baseline,
    find_label,
    remove_unicity,
    retrieve_longitudinal,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.iotools.iotools import commandline_to_json

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"

# coding: utf8


sex_dict = {"M": 0, "F": 1}
logger = getLogger("clinicadl.tsvtools.split")


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
        if p_val < p_min:
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


def make_split(
    data_tsv: Path,
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
    output_dir = _get_output_dir(data_tsv=data_tsv, output_dir=output_dir)
    output_dir = _get_results_directory(output_dir=output_dir)
    df = tsv_to_df(data_tsv)
    stratification = _get_stratification(
        df=df, ignore_demographics=ignore_demographics, stratification=stratification
    )

    baseline_df = extract_baseline(df)
    if n_test > 0:
        n_test = int(n_test) if n_test >= 1 else int(n_test * len(baseline_df))

        if not ignore_demographics:
            sex = list(baseline_df["sex"].values)
            age = list(baseline_df["age"].values)
            for i in stratification:
                category = list(baseline_df[i].values)
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

                    if p_sex >= p_sex_threshold and p_age >= p_age_threshold:
                        flag_selection = False
                        test_df = baseline_df.loc[test_index]
                        train_df = baseline_df.loc[train_index]

                    n_try += 1
            logger.info(f"Split was found after {n_try} trials.")

        else:
            idx = np.arange(len(baseline_df))
            idx_test = np.random.choice(idx, size=n_test, replace=False).sort()
            idx_train = complementary_list(idx, idx_test)

            test_df = baseline_df.loc[idx_test]
            train_df = baseline_df.loc[idx_train]

        name = f"{subset_name}_baseline.tsv"
        df_to_tsv(name, output_dir, test_df, baseline=True)

        if valid_longitudinal:
            name = f"{subset_name}.tsv"
            long_test_df = retrieve_longitudinal(test_df, df)
            df_to_tsv(name, output_dir, long_test_df)

    else:
        train_df = baseline_df

    name = "train_baseline.tsv"
    df_to_tsv(name, output_dir, train_df, baseline=True)

    long_train_df = retrieve_longitudinal(train_df, df)
    name = "train.tsv"
    df_to_tsv(name, output_dir, long_train_df)


def _get_results_directory(
    output_dir: Path, n_splits: Optional[NonNegativeInt] = None
) -> Path:
    split_numero = 1
    if n_splits:
        folder_name = f"{n_splits}_fold"
    else:
        folder_name = "split"

    while (output_dir / folder_name).is_dir():
        split_numero += 1
        folder_name = f"{folder_name}_{split_numero}"

    results_directory = output_dir / folder_name
    results_directory.mkdir(parents=True)

    return results_directory


def _get_stratification(
    df: pd.DataFrame,
    ignore_demographics: bool,
    stratification: Optional[List[str]] = None,
) -> Optional[List[str]]:
    list_columns = df.columns.values
    if stratification:
        if not set(stratification).issubset(set(list_columns)):
            raise ValueError(
                f"Stratification variable {stratification} does not exist in the dataset"
                "Your may want to give a tsv file with these columns and the missing information"
            )
        elif ignore_demographics:
            raise ValueError(
                "Stratification variable cannot be used when ignoring demographics"
            )

        elif not ignore_demographics and (
            "age" not in list_columns or "sex" not in list_columns
        ):
            raise ClinicaDLTSVError(
                "Your dataset doesn't contain one of these columns : age, sex, diagnosis"
                "Your may want to give a tsv file with these columns and the missing information"
                "You can also add the the flag --ignore_demographics to split without trying to balance age or sex distributions."
            )
        else:
            return stratification
    else:
        if ignore_demographics:
            return None
        else:
            return ["age", "sex"]


def _get_output_dir(data_tsv: Path, output_dir: Optional[Path]):
    if isinstance(data_tsv, str):
        data_tsv = Path(data_tsv)

    if not data_tsv.is_file():
        raise FileNotFoundError(f"The file {data_tsv} does not exist.")

    if not output_dir:
        output_dir = data_tsv.parent
    elif not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    return output_dir


def make_kfold(
    data_tsv: Path,
    output_dir: Optional[Path] = None,
    n_splits: NonNegativeInt = 5,
    subset_name: str = "validation",
    stratification: Optional[List[str]] = None,
    ignore_demographics: bool = False,
    valid_longitudinal: bool = False,
) -> Path:
    """
    Performs a K-fold split for cross-validation at the subject level. Each fold contains
    baseline and longitudinal train sets and a baseline test set.

    Parameters
    ----------
    data_tsv : Path
        Path to the input TSV file containing the dataset.
    output_dir : Optional[Path], default=None
        Path to the directory where split results will be saved.
        If None, results are saved in the parent directory of `data_tsv`.
    n_splits : NonNegativeInt, default=5
        Number of folds for the K-fold split.
    subset_name : str, default="validation"
        Name to use for the test subset.
    stratification : Optional[List[str]], default=None
        Columns to use for stratification. By default, age and sex are used unless
        `ignore_demographics` is set to True.
    ignore_demographics : bool, default=False
        If True, splits are performed without balancing demographic variables (age, sex).
    valid_longitudinal : bool, default=False
        If True, includes longitudinal data for the test set.

    Returns
    -------
    Path
        Path to the directory where the splits are saved.
    """
    # Validate and prepare output directory
    output_dir = _get_output_dir(data_tsv=data_tsv, output_dir=output_dir)
    output_dir = _get_results_directory(output_dir=output_dir, n_splits=n_splits)

    # Load the dataset and extract baseline sessions
    df = tsv_to_df(data_tsv)
    baseline_df = extract_baseline(df)

    # Define stratification criteria
    stratification = _get_stratification(
        df=df, ignore_demographics=ignore_demographics, stratification=stratification
    )

    # Create stratification labels or use uniform labels
    if stratification:  # TODO: improve with arya code
        stratification_list = list(baseline_df[stratification])
        unique = list(set(stratification_list))
        y = np.array([unique.index(x) for x in stratification_list])
    else:
        y = np.zeros(len(baseline_df))

    # Perform Stratified K-Fold splitting
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(y)), y)):
        # Split the data
        train_df = baseline_df.iloc[train_index]
        test_df = baseline_df.iloc[test_index]

        # Retrieve longitudinal data for training
        long_train_df = retrieve_longitudinal(train_df, df)

        # Reset indices for consistency
        train_df.reset_index(drop=True, inplace=True)
        long_train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        # Save the splits
        split_dir = output_dir / f"split-{i}"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(split_dir / "train_baseline.tsv", sep="\t", index=False)
        test_df.to_csv(split_dir / f"{subset_name}_baseline.tsv", sep="\t", index=False)
        long_train_df.to_csv(split_dir / "train.tsv", sep="\t", index=False)

        if valid_longitudinal:
            long_test_df = retrieve_longitudinal(test_df, df)
            long_test_df.reset_index(drop=True, inplace=True)
            long_test_df.to_csv(split_dir / f"{subset_name}.tsv", sep="\t", index=False)

    return output_dir
