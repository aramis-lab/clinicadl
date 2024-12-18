from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import PositiveFloat
from scipy.stats import chisquare, ks_2samp, ttest_ind
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from clinicadl.dataset.utils import tsv_to_df
from clinicadl.splitter.make_splits.utils import write_to_csv
from clinicadl.splitter.splitter.single_split import SingleSplitConfig
from clinicadl.tsvtools.tsvtools_utils import extract_baseline
from clinicadl.utils.exceptions import ClinicaDLConfigurationError, ClinicaDLTSVError

logger = getLogger("clinicadl.splitter.single_split")


def _validate_stratification(
    df: pd.DataFrame,
    stratification: Union[List[str], bool],
) -> List[str]:
    """
    Checks and validates the specified stratification columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    stratification : Union[List[str], bool]
        Columns to use for stratification. If True, columns are 'age' and 'sex', if False, there is no stratification.

    Returns
    -------
    List[str], optional
        Validated list of stratification columns or None if no stratification is applied.

    Raises
    ------
    ValueError
        If specified stratification columns are missing or if stratification conflicts with demographic handling.
    ClinicaDLTSVError
        If required demographic columns ('age', 'sex') are missing when not ignored.
    """

    if isinstance(stratification, bool):
        if stratification:
            stratification = ["age", "sex"]
        else:
            return []

    if isinstance(stratification, list):
        if not set(stratification).issubset(df.columns):
            raise ValueError(
                f"Invalid stratification columns: {set(stratification) - set(df.columns)}"
            )
        return stratification

    raise ValueError(
        "Invalid stratification option. Stratification must be a list of column names or a boolean."
    )


def _categorize_labels(
    df: pd.DataFrame,
    stratification: Union[List[str], bool],
    n_test: int = 100,
) -> Tuple[List[str], List[str]]:
    """
    Categorize stratification columns into continuous and categorical labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    stratification : Union[List[str], bool]
        Columns to use for stratification. If True, columns are 'age' and 'sex', if False, there is no stratification.
    n_test : int
        Number of test samples.

    Returns
    -------
    Tuple[List[str], List[str]]
        Continuous and categorical labels.
    """
    columns = _validate_stratification(df, stratification)

    continuous_labels, categorical_labels = [], []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() >= (n_test / 2):
            continuous_labels.append(col)
        else:
            categorical_labels.append(col)
    return continuous_labels, categorical_labels


def _chi2_test(x_test: List[int], x_train: List[int]) -> float:
    """
    Perform the Chi-squared test on categorical data.

    Parameters
    ----------
    x_test : np.ndarray
        Test data.
    x_train : np.ndarray
        Train data.

    Returns
    -------
    float
        p-value from the Chi-squared test.
    """
    unique_categories = np.unique(np.concatenate([x_test, x_train]))

    # Calculate observed (test) and expected (train) frequencies as raw counts
    f_obs = np.array([(x_test == category).sum() for category in unique_categories])
    f_exp = np.array(
        [
            (x_train == category).sum() / len(x_train) * len(x_test)
            for category in unique_categories
        ]
    )

    _, p_value = chisquare(f_obs, f_exp)

    return p_value


def make_split(
    tsv_path: Path,
    output_dir: Optional[Union[Path, str]] = None,
    n_test: PositiveFloat = 100,
    subset_name: str = "test",
    p_categorical_threshold: float = 0.50,
    p_continuous_threshold: float = 0.50,
    stratification: Union[List[str], bool] = False,
    valid_longitudinal=False,
    n_try_max: int = 1000,
):
    """
    Perform a single train-test split of the dataset with stratification.

    Parameters
    ----------
    tsv_path : Path
        Path to the input TSV file.
    output_dir : Optional[Path]
        Directory to save the split files.
    n_test : PositiveFloat
        If >= 1, specifies the absolute number of test samples. If < 1, treated as a proportion of the dataset.
    subset_name : str
        Name for the test subset.
    p_categorical_threshold : float
        Threshold for acceptable categorical stratification.
    p_continuous_threshold : float
        Threshold for acceptable continuous stratification.
    stratification : Union[List[str], bool], default=False
        Columns to use for stratification. If True, columns are 'age' and 'sex', if False, there is no stratification.
    valid_longitudinal : bool
        Include longitudinal sessions if True.
    n_try_max : int
        Maximum number of attempts to find a valid split.

    Returns
    -------
    Path
        Directory containing the split files.
    """

    # Set default output directory
    output_dir = output_dir or tsv_path.parent
    output_dir = Path(output_dir)

    # Load dataset and preprocess
    df = tsv_to_df(tsv_path)
    baseline_df = extract_baseline(df)

    n_test = int(n_test) if n_test >= 1 else int(n_test * len(baseline_df))

    continuous_labels, categorical_labels = _categorize_labels(
        df=baseline_df,
        stratification=stratification,
        n_test=n_test,
    )

    # Initialize SingleSplit configuration
    config = SingleSplitConfig(
        split_dir=output_dir,
        subset_name=subset_name,
        valid_longitudinal=valid_longitudinal,
        n_test=n_test,
        p_continuous_threshold=p_continuous_threshold,
        p_categorical_threshold=p_categorical_threshold,
        stratification=stratification,
    )

    config._check_split_dir()
    config._write_json()

    if config.n_test > 0:
        splits = ShuffleSplit(
            n_splits=n_try_max, test_size=config.n_test, random_state=2
        )
        for n_try, (train_index, test_index) in enumerate(
            splits.split(baseline_df, baseline_df)
        ):
            p_continuous = compute_continuous_p_value(
                continuous_labels,
                baseline_df,
                train_index.tolist(),
                test_index.tolist(),
            )

            if p_continuous >= p_continuous_threshold:
                p_categorical = compute_categorical_p_value(
                    categorical_labels,
                    baseline_df,
                    train_index.tolist(),
                    test_index.tolist(),
                )

                if p_categorical >= p_categorical_threshold:
                    logger.info(f"Valid split found after {n_try} attempts.")

                    test_df = baseline_df.loc[test_index]
                    train_df = baseline_df.loc[train_index]

                    write_continuous_stats(
                        config.split_dir / "split_continuous_stats.tsv",
                        continuous_labels,
                        test_df,
                        train_df,
                        subset_name,
                    )
                    write_categorical_stats(
                        config.split_dir / "split_categorical_stats.tsv",
                        categorical_labels,
                        test_df,
                        train_df,
                        baseline_df,
                        subset_name,
                    )
                    break

            if n_try >= n_try_max - 1:
                raise ClinicaDLConfigurationError(
                    f"Unable to find a valid split after {n_try} attempts. "
                    f"Consider lowering thresholds or reducing stratification variables."
                )

        write_to_csv(test_df, config.split_dir, df, subset_name, valid_longitudinal)
    else:
        train_df = baseline_df

    write_to_csv(train_df, config.split_dir, df)

    return config.split_dir


def compute_continuous_p_value(
    continuous_labels: Optional[list[str]],
    baseline_df: pd.DataFrame,
    train_index: list[int],
    test_index: list[int],
) -> float:
    """
    Compute the minimum p-value for continuous variables between train and test splits.

    Parameters
    ----------
    continuous_labels : Optional[List[str]]
        List of continuous variable names.
    baseline_df : pd.DataFrame
        Dataframe containing the baseline data.
    train_index : List[int]
        Indices for the training set.
    test_index : List[int]
        Indices for the testing set.

    Returns
    -------
    float
        The minimum p-value across all continuous labels.
    """

    p_continuous = 1.0
    if continuous_labels:
        for label in continuous_labels:
            if len(baseline_df[label] != 1):
                train_values = baseline_df[label].loc[train_index].values.tolist()
                test_values = baseline_df[label].loc[test_index].values.tolist()

                _, new_p_continuous = ttest_ind(
                    test_values, train_values, nan_policy="omit"
                )  # ks_2samp, or ttost_ind from statsmodels.stats.weightstats import ttost_ind

            # Track the minimum p-value
            p_continuous = min(p_continuous, new_p_continuous)

    return p_continuous


def compute_categorical_p_value(
    categorical_labels: Optional[list[str]],
    baseline_df: pd.DataFrame,
    train_index: list[int],
    test_index: list[int],
) -> float:
    """
    Compute the minimum p-value for categorical variables between train and test splits.

    Parameters
    ----------
    categorical_labels : Optional[List[str]]
        List of categorical variable names.
    baseline_df : pd.DataFrame
        Dataframe containing the baseline data.
    train_index : List[int]
        Indices for the training set.
    test_index : List[int]

    Returns
    -------
    float
        The minimum p-value across all categorical labels.
    """

    p_categorical = 1
    if categorical_labels:
        for label in categorical_labels:
            if len(baseline_df[label] != 1):
                mapping = {
                    val: i for i, val in enumerate(np.unique(baseline_df[label]))
                }

                tmp_train_values = baseline_df[label].loc[train_index].values.tolist()
                tmp_test_values = baseline_df[label].loc[test_index].values.tolist()

                train_values = [mapping[val] for val in tmp_train_values]
                test_values = [mapping[val] for val in tmp_test_values]

                new_p_categorical = _chi2_test(test_values, train_values)

            # Track the minimum p-value
            p_categorical = min(p_categorical, new_p_categorical)

    return p_categorical


def write_continuous_stats(
    tsv_path: Path,
    continuous_labels: Optional[list[str]],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    subset_name: str,
):
    """
    Write continuous statistics (mean, std) to a TSV file.

    Parameters
    ----------
    tsv_path : Path
        Path to save the output TSV file.
    continuous_labels : Optional[List[str]]
        List of continuous variable names.
    test_df : pd.DataFrame
        Test dataset.
    train_df : pd.DataFrame
        Train dataset.
    subset_name : str
        Name of the test subset.
    """

    if not continuous_labels:
        return

    data = [
        (label, "mean", train_df[label].mean(), test_df[label].mean())
        for label in continuous_labels
    ] + [
        (label, "std", train_df[label].std(), test_df[label].std())
        for label in continuous_labels
    ]

    df_stats_continuous = pd.DataFrame(
        data, columns=["label", "statistic", "train", subset_name]
    )
    df_stats_continuous.to_csv(tsv_path, sep="\t", index=False)


def write_categorical_stats(
    tsv_path: Path,
    categorical_labels: Optional[list[str]],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    subset_name: str,
):
    """
    Write categorical statistics (proportion, count) to a TSV file.

    Parameters
    ----------
    tsv_path : Path
        Path to save the output TSV file.
    categorical_labels : Optional[List[str]]
        List of categorical variable names.
    test_df : pd.DataFrame
        Test dataset.
    train_df : pd.DataFrame
        Train dataset.
    baseline_df : pd.DataFrame
        Baseline dataset (reference for all unique values).
    subset_name : str
        Name

    """

    if not categorical_labels:
        return

    data = []
    for label in categorical_labels:
        unique_values = baseline_df[label].unique()
        for val in unique_values:
            test_count = (test_df[label] == val).sum()
            train_count = (train_df[label] == val).sum()

            test_proportion = test_count / len(test_df)
            train_proportion = train_count / len(train_df)

            data.append((label, val, "proportion", train_proportion, test_proportion))
            data.append((label, val, "count", train_count, test_count))

    df_stats_categorical = pd.DataFrame(
        data, columns=["label", "value", "statistic", "train", subset_name]
    )
    df_stats_categorical.to_csv(tsv_path, sep="\t", index=False)
