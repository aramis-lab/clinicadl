from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from clinicadl.dictionary.words import FOLD, VALIDATION
from clinicadl.split.splitter.kfold import KFoldConfig
from clinicadl.utils.typing import DataType, PathType

from .utils import (
    extract_baseline,
    find_available_split_dir,
    read_and_format_data,
    write_to_tsv,
)


def make_kfold(
    data: DataType,
    n_splits: int = 5,
    output_dir: Optional[PathType] = None,
    subset_name: str = VALIDATION,
    stratification: Union[str, bool] = False,
    longitudinal: bool = False,
    seed: Optional[int] = None,
) -> Path:
    r"""
    Performs K-Fold splitting on a DataFrame with optional stratification.

    Stratification can be performed based on a **categorical** variable of the DataFrame.

    .. note::
        ``make_kfold`` splits the **participants** in your data. This means that, if all the participants don't have the
        same number of sessions, you may likely end up with training sets of different sizes across your splits.
        Besides, by default, only one session per participant is kept in the validation sets (see the argument
        ``longitudinal``).

    Parameters
    ----------
    data: Union[pd.DataFrame, Path, str],
        A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of participant/session
        pairs to split.
    n_splits : int, (optional, default=5)
        Number of folds. Must be at least 2.
    output_dir : Optional[Path, str], (optional, default=None)
        Directory where to save the output files of the split, passed as a ``str`` or a :pathlib.Path:`pathlib.Path <>`.
        If ``data`` is a path and ``output_dir`` is not passed, the parent directory of the TSV file will be used.
    subset_name : str, (optional, default="validation")
        Name for the validation subset.
    stratification : Union[str, bool], (optional, default=False)
        Whether to perform stratification. If ``True``, the columns ``"sex"`` will be used for stratification.
        If a ``str`` is passed, this column will be used. The variable associated to the column must be
        **categorical**.
    longitudinal : bool, (optional, default=False)
        Whether to include only the baseline sessions in the validation set (``longitudinal=False``). If ``True``, all the sessions
        of the validation participants will be included. No matter this argument, all sessions are always kept in the training set.
    seed : Optional[int], (optional, default=None)
        Seed to control the randomness of the split. Useful for reproducibility.

    Returns
    -------
    Path
        Directory containing the generated split files.

    Raises
    ------
    ValueError
        If ``data`` is a :py:class:`pandas.DataFrame` and no ``output_dir`` is passed.
    ClinicaDLTSVError
        If the DataFrame does not contain the columns ``"participant_id"`` and ``"session_id"``.
    KeyError
        If the stratification column mentioned via ``stratification`` cannot be found in the DataFrame.
    ValueError
        If the stratification column mentioned via ``stratification`` is not a categorical variable.

    See Also
    --------
    - :py:func:`~clinicadl.split.make_split`

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

    >>> from clinicadl.split import make_kfold
    >>> split_dir = make_kfold(
            df,
            n_splits=5,
            output_dir="splits",
            stratification="sex",
        )
    >>> split_dir
    PosixPath('splits/5_fold')
    # splits/5_fold
    # ├── kfold_config.json
    # ├── split-0
    # │   ├── train.tsv
    # │   ├── train_baseline.tsv
    # │   └── validation_baseline.tsv
    # ├── split-1
    # │   └── ...
    # ├── split-2
    # │   └── ...
    # ├── split-3
    # │   └── ...
    # └── split-4
    #     └── ...

    >>> train_baseline = pd.read_csv(split_dir / "split-0" / "train_baseline.tsv", sep="\t")
    >>> train_baseline.head(5)
        participant_id	session_id	sex
    0	sub-003	        ses-M000	M
    1	sub-005	        ses-M006	F
    2	sub-006	        ses-M006	M
    3	sub-014	        ses-M000	M
    4	sub-015	        ses-M006	F
    >>> len(train_baseline)
    32
    >>> val_baseline = pd.read_csv(split_dir / "split-0" / "validation_baseline.tsv", sep="\t")
    >>> val_baseline.head(5)
        participant_id	session_id	sex
    0	sub-004	        ses-M000	M
    1	sub-007	        ses-M006	F
    2	sub-013	        ses-M000	M
    3	sub-023	        ses-M000	M
    4	sub-026	        ses-M006	M
    >>> len(val_baseline)
    8
    """
    df = read_and_format_data(data)

    if isinstance(data, (str, Path)):
        output_dir = output_dir or Path(data).parent
    elif isinstance(data, pd.DataFrame) and not output_dir:
        raise ValueError("You must specify the output directory.")
    output_dir = Path(output_dir)

    _stratification = _validate_stratification(df, stratification)

    kfold_dir = find_available_split_dir(output_dir, f"{n_splits}_{FOLD}")
    config = KFoldConfig(
        split_dir=kfold_dir,
        subset_name=subset_name,
        longitudinal=longitudinal,
        n_splits=n_splits,
        stratification=_stratification,
        seed=seed,
    )

    baseline_df = extract_baseline(
        df, columns=[config.stratification] if config.stratification else None
    )
    stratifying_labels = (
        baseline_df[config.stratification] if config.stratification else None
    )

    if config.stratification:
        skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=seed)
    else:
        skf = KFold(n_splits=config.n_splits, shuffle=True, random_state=seed)

    for i, (train_idx, val_idx) in enumerate(
        skf.split(baseline_df, stratifying_labels)
    ):
        train_df = baseline_df.iloc[train_idx]
        val_df = baseline_df.iloc[val_idx]

        split_subdir = config.get_split_subdir(i, create=True)

        write_to_tsv(val_df, split_subdir, config.subset_name, df, config.longitudinal)
        write_to_tsv(
            train_df, split_subdir, config._training_subset_name, df, longitudinal=True
        )

    config.write_json()

    return kfold_dir


def _validate_stratification(
    df: pd.DataFrame,
    stratification: Union[str, bool],
) -> Optional[str]:
    """
    Validates and checks the stratification columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    stratification : Union[str, bool]
        Column to use for stratification. If True, column is 'sex', if False, there is no stratification.

    Returns
    -------
    Optional[str]
        Validated stratification column or None if no stratification is applied.
    """
    if isinstance(stratification, bool):
        if stratification:
            stratification = "sex"
        else:
            return None

    if isinstance(stratification, List):
        if len(stratification) > 1:
            raise ValueError(
                f"Stratification can only be performed on a single column for K-Fold splitting. Got: {stratification}"
            )
        else:
            stratification = stratification[0]

    if isinstance(stratification, str):
        if stratification not in df.columns:
            raise KeyError(
                f"Stratification column '{stratification}' not found in the dataset."
            )

        if pd.api.types.is_numeric_dtype(df[stratification]) and df[
            stratification
        ].nunique() >= (len(df) / 2):
            raise ValueError(
                "Continuous variables cannot be used for stratification in K-Fold splitting."
            )
        return stratification

    raise ValueError(
        f"Invalid stratification option. Stratification must be a single column name or a boolean. Got: {stratification}"
    )
