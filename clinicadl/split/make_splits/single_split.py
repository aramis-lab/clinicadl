from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ttest_ind
from sklearn.model_selection import ShuffleSplit

from clinicadl.dictionary.words import (
    AGE,
    COUNT,
    LABEL,
    MEAN,
    PROPORTION,
    SEX,
    SPLIT,
    STATISTIC,
    STD,
    TEST,
    TRAIN,
    VALUE,
)
from clinicadl.split.splitter.single_split import SingleSplitConfig
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import DataType, PathType

from .utils import (
    extract_baseline,
    find_available_split_dir,
    read_and_format_data,
    write_to_tsv,
)

logger = getLogger("clinicadl.split.make_splits.single_split")


def make_split(
    data: DataType,
    n_test: float = 0.2,
    output_dir: Optional[PathType] = None,
    subset_name: str = TEST,
    stratification: Union[List[str], bool] = False,
    p_categorical_threshold: float = 0.80,
    p_continuous_threshold: float = 0.80,
    longitudinal: bool = False,
    n_try_max: int = 1000,
    seed: Optional[int] = None,
) -> Path:
    r"""
    Performs a single train-test split on a DataFrame with optional stratification.

    Stratification can be performed based on one or several variables present in the DataFrame:

    - If a variable is **categorical**, a `chi-squared test <https://en.wikipedia.org/wiki/Chi-squared_test>`_
      is performed to check that the train and test sets have the same distribution.
    - If a variable is **continuous**, a `t-test <https://en.wikipedia.org/wiki/Student%27s_t-test>`_ is performed.

    ``make_split`` will try random splits until one split shows a p-values greater than ``p_categorical_threshold`` for all
    categorical variables used for stratification, and greater than ``p_continuous_threshold`` for all continuous variables.
    So, ``p_categorical_threshold`` and ``p_continuous_threshold`` controls the required level of similarity between the train
    and the test distributions. The higher the threshold, the more demanding the similarity test. So, too high a threshold may
    prevent you from finding a valid split.

    Parameters
    ----------
    data: Union[pd.DataFrame, Path, str],
        A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of participant/session
        pairs to split.
    n_test : PositiveFloat, (optional, default=0.2)
        A positive float. If ``>=1``, it specifies the number of test participants. If ``>1``, it is treated as a proportion of all
        participants to have in the test data.

        .. note::
            Here, we are talking about number of **participants**. So, if ``n_test=0.2``, it doesn't mean that you have 80%
            of your data in the training set, but rather that you have 80% of you participants in the training set.

    output_dir : Optional[Union[Path, str]], (optional, default=None)
        Directory where to save the output files of the split, passed as a ``str`` or a :pathlib.Path:`pathlib.Path <>`.
        If ``data`` is a path and ``output_dir`` is not passed, the parent directory of the TSV file will be used.
    subset_name : str, (optional, default="test")
        Name for the test subset.
    stratification : Union[List[str], bool], (optional, default=False)
        Whether to perform stratification. If ``True``, the columns ``"age"`` and ``"sex"`` will be used for stratification.
        If a list of ``str`` is passed, these columns will be used.
    p_categorical_threshold : float, (optional, default=0.80)
        Threshold for acceptable categorical stratification. Must be **between 0 and 1**.
    p_continuous_threshold : float, (optional, default=0.80)
        Threshold for acceptable continuous stratification. Must be **between 0 and 1**.
    longitudinal : bool, (optional, default=False)
        Whether to include only the baseline sessions in the test set (``longitudinal=False``). If ``True``, all the sessions
        of the test participants will be included. No matter this argument, all sessions are always kept in the training set.
    n_try_max : int, (optional, default=1000)
        Maximum number of attempts to find a valid split.
    seed : Optional[int], (optional, default=None)
        Seed to control the randomness of the split. Useful for reproducibility.

    Returns
    -------
    Path
        Directory containing the split files.

    Raises
    ------
    ValueError
        If ``data`` is a :py:class:`pandas.DataFrame` and no ``output_dir`` is passed.
    ClinicaDLTSVError
        If the DataFrame does not contain the columns ``"participant_id"`` and ``"session_id"``.
    KeyError
        If the stratification columns mentioned via ``stratification`` cannot be found in the DataFrame.
    ClinicaDLConfigurationError
        If no good split was found after ``n_try_max`` tries.

    See Also
    --------
    - :py:func:`~clinicadl.split.make_kfold`

    Examples
    --------
    >>> df.head(5)  # quick look at the data
        participant_id	session_id	age	sex	diagnosis
    0	sub-003	        ses-M000	40	M	MCI
    1	sub-004	        ses-M000	56	M	CN
    2	sub-004	        ses-M054	75	F	MCI
    3	sub-005	        ses-M006	85	F	CN
    4	sub-005	        ses-M018	64	M	AD
    >>> len(df)
    64

    >>> from clinicadl.split import make_split
    >>> split_dir = make_split(
            df,
            output_dir="splits",
            stratification=["sex", "age"],
            p_categorical_threshold=0.9,
            p_continuous_threshold=0.9,
        )
    >>> split_dir
    PosixPath('splits/split')
    # splits/split
    # ├── single_split_config.json
    # ├── split_categorical_stats.tsv
    # ├── split_continuous_stats.tsv
    # ├── test_baseline.tsv
    # ├── train.tsv
    # └── train_baseline.tsv

    >>> pd.read_csv(split_dir / "train.tsv", sep="\t").head(5)
        participant_id	session_id
    0	sub-005	        ses-M006
    1	sub-005	        ses-M018
    2	sub-065	        ses-M006
    3	sub-065	        ses-M018
    4	sub-044	        ses-M000
    >>> train_baseline = pd.read_csv(split_dir / "train_baseline.tsv", sep="\t")
    >>> train_baseline.head(5)
        participant_id	session_id	sex	age
    0	sub-005	        ses-M006	F	85
    1	sub-065	        ses-M006	F	58
    2	sub-044	        ses-M000	M	64
    3	sub-014	        ses-M000	M	56
    4	sub-043	        ses-M000	M	71
    >>> len(train_baseline)
    32
    >>> test_baseline = pd.read_csv(split_dir / "test_baseline.tsv", sep="\t")
    >>> test_baseline.head(5)
        participant_id	session_id	sex	age
    0	sub-037	        ses-M006	F	56
    1	sub-003	        ses-M000	M	40
    2	sub-077	        ses-M006	F	23
    3	sub-023	        ses-M000	M	69
    4	sub-057	        ses-M006	F	77
    >>> len(test_baseline)
    8

    >>> pd.read_csv(split_dir / "split_continuous_stats.tsv", sep="\t")
        label	statistic	train	test
    0	age	    mean	62.8	63.6
    1	age	    std	        18.0	22.4
    >>> pd.read_csv(split_dir / "split_categorical_stats.tsv", sep="\t")
        label	value	statistic	train	test
    0	sex	    F	    proportion	0.41	0.38
    1	sex	    F	    count	13.0	3.0
    2	sex	    M	    proportion	0.59	0.62
    3	sex	    M	    count	19.0	5.0

    """
    df = read_and_format_data(data)

    if isinstance(data, (str, Path)):
        output_dir = output_dir or Path(data).parent
    elif isinstance(data, pd.DataFrame) and not output_dir:
        raise ValueError("You must specify the output directory.")
    output_dir = Path(output_dir)

    stratification = _validate_stratification(df, stratification)
    baseline_df = extract_baseline(df, columns=stratification)

    split_dir = find_available_split_dir(output_dir, SPLIT)
    config = SingleSplitConfig(
        split_dir=split_dir,
        subset_name=subset_name,
        longitudinal=longitudinal,
        n_test=n_test,
        p_continuous_threshold=p_continuous_threshold,
        p_categorical_threshold=p_categorical_threshold,
        stratification=stratification,
        seed=seed,
    )

    n_test = int(n_test) if n_test >= 1 else int(n_test * len(baseline_df))

    continuous_labels, categorical_labels = _categorize_labels(
        df=baseline_df,
        stratification=config.stratification,
        n_test=n_test,
    )

    if n_test == 0:
        train_df = baseline_df
        test_df = pd.DataFrame(columns=train_df.columns)

    else:
        splits = ShuffleSplit(n_splits=n_try_max, test_size=n_test, random_state=seed)
        for n_try, (train_index, test_index) in enumerate(
            splits.split(baseline_df), start=1
        ):
            p_continuous = _compute_continuous_p_value(
                continuous_labels,
                baseline_df,
                train_index.tolist(),
                test_index.tolist(),
            )

            if p_continuous >= p_continuous_threshold:
                p_categorical = _compute_categorical_p_value(
                    categorical_labels,
                    baseline_df,
                    train_index.tolist(),
                    test_index.tolist(),
                )

                if p_categorical >= p_categorical_threshold:
                    logger.info("Valid split found after %f attempts.", n_try)

                    test_df = baseline_df.loc[test_index]
                    train_df = baseline_df.loc[train_index]

                    _write_continuous_stats(
                        split_dir / "split_continuous_stats.tsv",
                        continuous_labels,
                        test_df,
                        train_df,
                        subset_name,
                    )
                    _write_categorical_stats(
                        split_dir / "split_categorical_stats.tsv",
                        categorical_labels,
                        test_df,
                        train_df,
                        subset_name,
                    )
                    break

        else:
            raise ClinicaDLConfigurationError(
                f"Unable to find a valid split after {n_try_max} attempts. "
                "Consider lowering thresholds or removing some stratification variables."
            )

    write_to_tsv(test_df, split_dir, subset_name, df, longitudinal)
    write_to_tsv(
        train_df, split_dir, config._training_subset_name, df, longitudinal=True
    )
    config.write_json()

    return split_dir


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
    """

    if isinstance(stratification, bool):
        if stratification:
            stratification = [AGE, SEX]
        else:
            return []

    if isinstance(stratification, list):
        if not set(stratification).issubset(df.columns):
            raise KeyError(
                f"Invalid stratification columns (not found in the dataframe): {set(stratification) - set(df.columns)}"
            )
        return stratification

    raise ValueError(
        f"Invalid stratification option. Stratification must be a list of column names or a boolean. Got: {stratification}"
    )


def _categorize_labels(
    df: pd.DataFrame,
    stratification: List[str],
    n_test: int,
) -> Tuple[List[str], List[str]]:
    """
    Categorize stratification columns into continuous and categorical labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    stratification : List[str]
        Columns to use for stratification.
    n_test : int
        Number of test samples.

    Returns
    -------
    Tuple[List[str], List[str]]
        Continuous and categorical labels.
    """
    continuous_labels, categorical_labels = [], []
    for col in stratification:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() >= (n_test / 2):
            continuous_labels.append(col)
        else:
            categorical_labels.append(col)
    return continuous_labels, categorical_labels


def _compute_continuous_p_value(
    continuous_labels: List[str],
    baseline_df: pd.DataFrame,
    train_index: list[int],
    test_index: list[int],
) -> float:
    """
    Compute the minimum p-value for continuous variables between train and test splits.

    Parameters
    ----------
    continuous_labels : List[str]
        List of continuous variable names (can be empty).
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
            train_values = baseline_df.loc[train_index, label]
            test_values = baseline_df.loc[test_index, label]

            _, new_p_continuous = ttest_ind(
                test_values.tolist(), train_values.tolist(), nan_policy="omit"
            )  # ks_2samp, or ttost_ind from statsmodels.stats.weightstats import ttost_ind

            if np.isnan(new_p_continuous):
                return 0.0  # can't compute the p-value so we won't choose this split

            p_continuous = min(p_continuous, new_p_continuous)

    return p_continuous


def _compute_categorical_p_value(
    categorical_labels: list[str],
    baseline_df: pd.DataFrame,
    train_index: list[int],
    test_index: list[int],
) -> float:
    """
    Compute the minimum p-value for categorical variables between train and test splits.

    Parameters
    ----------
    categorical_labels : List[str]
        List of categorical variable names (can be empty).
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

    p_categorical = 1.0
    if categorical_labels:
        for label in categorical_labels:
            mapping = {
                val: i for i, val in enumerate(np.unique(baseline_df[label].dropna()))
            }

            train_values = baseline_df.loc[train_index, label].apply(
                lambda val: mapping[val]
            )
            test_values = baseline_df.loc[test_index, label].apply(
                lambda val: mapping[val]
            )

            new_p_categorical = _chi2_test(test_values, train_values)

            p_categorical = min(p_categorical, new_p_categorical)

    return p_categorical


def _chi2_test(x_test: np.ndarray, x_train: np.ndarray) -> float:
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
    unique_categories = unique_categories[~np.isnan(unique_categories)]

    # Calculate observed (test) and expected (train) frequencies as raw counts
    f_obs = np.array([(x_test == category).sum() for category in unique_categories])
    f_obs = f_obs / np.sum(f_obs)
    f_exp = np.array([(x_train == category).sum() for category in unique_categories])
    f_exp = f_exp / np.sum(f_exp)

    _, p_value = chisquare(f_obs, f_exp)

    return p_value


def _write_continuous_stats(
    tsv_path: Path,
    continuous_labels: list[str],
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
    continuous_labels : List[str]
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
        (label, MEAN, train_df[label].mean(), test_df[label].mean())
        for label in continuous_labels
    ] + [
        (label, STD, train_df[label].std(), test_df[label].std())
        for label in continuous_labels
    ]

    df_stats_continuous = pd.DataFrame(
        data, columns=[LABEL, STATISTIC, TRAIN, subset_name]
    )
    df_stats_continuous.to_csv(tsv_path, sep="\t", index=False)


def _write_categorical_stats(
    tsv_path: Path,
    categorical_labels: list[str],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    subset_name: str,
):
    """
    Write categorical statistics (proportion, count) to a TSV file.

    Parameters
    ----------
    tsv_path : Path
        Path to save the output TSV file.
    categorical_labels : List[str]
        List of categorical variable names.
    test_df : pd.DataFrame
        Test dataset.
    train_df : pd.DataFrame
        Train dataset.
    subset_name : str
        Name of the test subset.
    """

    if not categorical_labels:
        return

    data = []
    for label in categorical_labels:
        unique_values = pd.concat([train_df, test_df])[label].unique()
        for value in unique_values:
            test_count = int((test_df[label] == value).sum())
            train_count = int((train_df[label] == value).sum())

            test_proportion = test_count / len(test_df)
            train_proportion = train_count / len(train_df)

            data.append((label, value, PROPORTION, train_proportion, test_proportion))
            data.append((label, value, COUNT, train_count, test_count))

    df_stats_categorical = pd.DataFrame(
        data, columns=[LABEL, VALUE, STATISTIC, TRAIN, subset_name]
    )
    df_stats_categorical.to_csv(tsv_path, sep="\t", index=False)
