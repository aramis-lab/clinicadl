from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import PositiveInt
from sklearn.model_selection import KFold, StratifiedKFold

from clinicadl.dataset.utils import tsv_to_df
from clinicadl.splitter.make_splits.utils import write_to_csv
from clinicadl.splitter.splitter.kfold import KFoldConfig
from clinicadl.tsvtools.tsvtools_utils import extract_baseline
from clinicadl.utils.exceptions import ClinicaDLConfigurationError


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

    Raises
    ------
    ClinicaDLConfigurationError
        If invalid or conflicting stratification options are provided.
    """
    if isinstance(stratification, bool):
        if stratification:
            stratification = "sex"
        else:
            return None

    if isinstance(stratification, List):
        if len(stratification) > 1:
            raise ClinicaDLConfigurationError(
                "Stratification can only be performed on a single column for K-Fold splitting."
            )
        else:
            stratification = stratification[0]

    if isinstance(stratification, str):
        if stratification not in df.columns:
            raise ClinicaDLConfigurationError(
                f"Stratification column '{stratification}' not found in the dataset."
            )

        if pd.api.types.is_numeric_dtype(df[stratification]) and df[
            stratification
        ].nunique() >= (len(df) / 2):
            raise ValueError(
                "Continuous variables cannot be used for stratification in K-Fold splitting."
            )
        return stratification

    raise ClinicaDLConfigurationError(
        "Invalid or conflicting stratification options provided. Stratification must be a single column name or boolean."
    )


def preprocess_stratification(
    df: pd.DataFrame,
    stratification: Union[str, bool],
) -> pd.DataFrame:
    """
    Preprocess stratification columns by creating labels for each subject.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    stratification : Union[str, bool]
        Column to use for stratification. If True, column is 'sex', if False, there is no stratification.

    Returns
    -------
    List[str]
        List of stratification labels for the dataset.
    """
    column = _validate_stratification(df, stratification)

    if column is None:
        return df

    return df[[column]]


def make_kfold(
    tsv_path: Path,
    output_dir: Optional[Union[Path, str]] = None,
    subset_name: str = "validation",
    valid_longitudinal: bool = False,
    n_splits: PositiveInt = 5,
    stratification: Union[str, bool] = False,
) -> Path:
    """
    Perform K-Fold splitting with optional stratification.

    Parameters
    ----------
    tsv_path : Path
        Path to the input TSV file.
    output_dir : Optional[Path]
        Directory to save the split files. Defaults to the parent directory of `tsv_path`.
    subset_name : str, default="validation"
        Name of the subset used for output files.
    valid_longitudinal : bool, default=False
        Whether to include longitudinal sessions in the split.
    n_splits : PositiveInt, default=5
        Number of splits for K-Fold.
    stratification : Union[str, bool], default=False
        Column to use for stratification. If True, column is 'sex', if False, there is no stratification.

    Returns
    -------
    Path
        Directory containing the generated split files.

    Raises
    ------
    ClinicaDLConfigurationError
        If invalid configuration options are provided.
    """

    # Set default output directory
    output_dir = output_dir or tsv_path.parent
    output_dir = Path(output_dir)

    # Initialize KFold configuration
    config = KFoldConfig(
        split_dir=output_dir,
        subset_name=subset_name,
        valid_longitudinal=valid_longitudinal,
        n_splits=n_splits,
        stratification=stratification,
    )

    config._check_split_dir()
    config._write_json()

    # Load and process dataset
    df = tsv_to_df(tsv_path)
    baseline_df = extract_baseline(df)

    stratify_labels = preprocess_stratification(
        df=baseline_df,
        stratification=config.stratification,
    )

    # Create K-Fold splits
    if config.stratification:
        skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=2)
    else:
        skf = KFold(n_splits=config.n_splits, shuffle=True, random_state=2)

    for i, (train_idx, test_idx) in enumerate(skf.split(baseline_df, stratify_labels)):
        train = baseline_df.iloc[train_idx]
        test = baseline_df.iloc[test_idx]

        split_dir = config.split_dir / f"split-{i}"
        split_dir.mkdir(parents=True, exist_ok=True)

        write_to_csv(test, split_dir, df, config.subset_name, config.valid_longitudinal)
        write_to_csv(train, split_dir, df)

    return config.split_dir
