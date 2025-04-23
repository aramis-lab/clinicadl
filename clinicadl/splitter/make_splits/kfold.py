from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pydantic import PositiveInt
from sklearn.model_selection import KFold, StratifiedKFold

from clinicadl.dictionary.words import FOLD
from clinicadl.splitter.splitter.kfold import KFoldConfig
from clinicadl.utils.typing import DataType, PathType

from .utils import (
    extract_baseline,
    find_available_split_dir,
    read_and_format_data,
    write_to_tsv,
)


def make_kfold(
    data: DataType,
    n_splits: PositiveInt = 5,
    output_dir: Optional[PathType] = None,
    subset_name: str = "validation",
    stratification: Union[str, bool] = False,
    longitudinal: bool = False,
    seed: Optional[int] = None,
) -> Path:
    """
    Perform K-Fold splitting with optional stratification.

    Stratification can be performed based on a **categorical** variable of the DataFrame.

    .. note::
        ``make_kfold`` splits the **participants** in your data. This means that, if all the participants don't have the
        same number of sessions, you may likely end up with training/validation sets of different sizes across your folds.
        Besides, by default, only one session per participant is kept in the validation sets (see ``longitudinal``).

    Parameters
    ----------
    data: Union[pd.DataFrame, Path, str],
        A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of participant/session
        pairs to split.
    n_splits : PositiveInt, (optional, default=5)
        Number of folds.
    output_dir : Optional[Path, str], (optional, default=None)
        Directory where to save the output files of the split. If ``data`` is a path and ``output_dir`` is not passed,
        the parent directory of the DataFrame will be used.
    subset_name : str, (optional, default="validation")
        Name for the validation subset.
    stratification : Union[str, bool], (optional, default=False)
        Whether to perform stratification. If ``True``, the columns ``sex`` will be used for stratification.
        If a ``str`` is passed, this column will be used. The variable associated to the column must be
        **categorical**.
    longitudinal : bool, (optional, default=False)
        Whether to include only the baseline sessions in the validation data (``longitudinal=False``). If ``True``, all the sessions
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
        If ``data`` is a DataFrame and no ``output_dir`` is passed.
    ClinicaDLTSVError
        If the required columns ('participant_id', 'session_id') are not found in the DataFrame.
    KeyError
        If the stratification column mentioned via ``stratification`` cannot be found in the DataFrame.
    ValueError
        If the stratification column mentioned via ``stratification`` is not a categorical variable.
    """
    df = read_and_format_data(data)

    if isinstance(data, (str, Path)):
        output_dir = output_dir or data.parent
    elif isinstance(data, pd.DataFrame) and not output_dir:
        raise ValueError("You must specify the output directory.")
    output_dir = Path(output_dir)

    stratification = _validate_stratification(df, stratification)

    split_dir = find_available_split_dir(output_dir, f"{n_splits}_{FOLD}")
    config = KFoldConfig(
        split_dir=split_dir,
        subset_name=subset_name,
        longitudinal=longitudinal,
        n_splits=n_splits,
        stratification=stratification,
    )

    baseline_df = extract_baseline(
        df, columns=[config.stratification] if config.stratification else None
    )
    stratifying_labels = (
        baseline_df[config.stratification] if config.stratification else None
    )

    # Create K-Fold splits
    if config.stratification:
        skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=seed)
    else:
        skf = KFold(n_splits=config.n_splits, shuffle=True, random_state=seed)

    for i, (train_idx, val_idx) in enumerate(
        skf.split(baseline_df, stratifying_labels)
    ):
        train_df = baseline_df.iloc[train_idx]
        val_df = baseline_df.iloc[val_idx]

        split_dir = config.get_fold_dir(i)

        write_to_tsv(val_df, split_dir, config.subset_name, df, config.longitudinal)
        write_to_tsv(
            train_df, split_dir, config._training_subset_name, df, longitudinal=True
        )

    config.write_json()

    return config.split_dir


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
